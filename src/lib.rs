#![allow(
    dead_code,
    unused_variables,
    unused_imports,
    unused_mut,
    unreachable_code
)]
#![feature(
    alloc_layout_extra,
    map_try_insert,
    core_intrinsics,
    const_trait_impl,
    const_type_id,
    const_mut_refs
)]
use std::alloc::{alloc, dealloc, realloc};
use std::collections::hash_map;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::ops::{Add, Index, IndexMut, Range, RangeBounds};
use std::ptr::copy;
use std::slice::SliceIndex;
use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    collections::HashMap,
    fmt::Debug,
};

// TODO: check to properly drop all manually init memory

const GENERATION_COMPTYPE: CompType = CompType::new::<Generation>();
const SPARSE_INDEX_COMPTYPE: CompType = CompType::new::<SparseIndex>();

//-----------------COMPONENT TYPE-----------------//
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompType {
    pub(crate) type_id: TypeId,
    pub(crate) layout: Layout,
}
impl CompType {
    pub(crate) const fn new<C: 'static>() -> Self {
        Self {
            type_id: TypeId::of::<C>(),
            layout: Layout::new::<C>(),
        }
    }
}

pub trait Component: Clone + 'static {
    fn get_id() -> CompType {
        CompType {
            type_id: TypeId::of::<Self>(),
            layout: Layout::new::<Self>(),
        }
    }
}

//-----------------TYPE ERASED POINTER-----------------//
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
// this is for content, not generation or sparse index
pub(crate) struct Ptr {
    pub(crate) ptr: *mut u8,
    pub(crate) comp_type: CompType,
    pub(crate) sparse_index: SparseIndex,
}
impl Ptr {
    pub(crate) fn new(ptr: *mut u8, comp_type: CompType, sparse_index: SparseIndex) -> Self {
        Self {
            ptr,
            comp_type,
            sparse_index,
        }
    }

    pub(crate) fn cast_value(self) -> Value {
        Value::new(self.ptr, self.comp_type)
    }

    pub(crate) unsafe fn cast<T: 'static>(&self) -> Result<&mut T, &'static str> {
        if CompType::new::<T>() != self.comp_type {
            return Err("type not matching");
        } else {
            Ok(self.ptr.cast::<T>().as_mut().ok_or("casting failure")?)
        }
    }
}

impl PartialOrd for Ptr {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.sparse_index.partial_cmp(&other.sparse_index)
    }
}

impl Ord for Ptr {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sparse_index.cmp(&other.sparse_index)
    }
}

//-----------------TYPE ERASED VALUE-----------------//
#[derive(Clone)]
pub(crate) struct Value {
    pub(crate) ptr: *mut u8,
    pub(crate) comp_type: CompType,
}
impl Value {
    pub(crate) fn new(src_ptr: *mut u8, comp_type: CompType) -> Self {
        unsafe {
            let ptr = alloc(comp_type.layout);
            std::ptr::copy(src_ptr, ptr, comp_type.layout.size());
            Self { ptr, comp_type }
        }
    }

    pub(crate) unsafe fn cast<T: 'static + Clone>(self) -> Result<T, &'static str> {
        if self.comp_type != CompType::new::<T>() {
            return Err("type not matching");
        } else {
            Ok(self.ptr.cast::<T>().as_ref().unwrap().clone())
        }
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr, self.comp_type.layout);
        }
    }
}

//-----------------TYPE ERASED PTR COLUMN/ROW-----------------//
// all with the same component type and different sparse index
#[derive(Clone)]
pub struct PtrColumn {
    pub(crate) comp_type: CompType,
    // sorted with sparse index
    pub(crate) vec: Vec<Ptr>,
}
impl PtrColumn {
    pub(crate) fn new_empty(comp_type: CompType) -> Self {
        Self {
            vec: vec![],
            comp_type,
        }
    }

    pub(crate) fn push(&mut self, mut access: Vec<Ptr>) -> Result<(), &'static str> {
        // right now it doesn't check if there's duplicate ptr sparse index
        for each in access.iter() {
            if each.comp_type != self.comp_type {
                return Err("type not matching");
            }
        }
        self.vec.append(&mut access);
        self.vec.sort();
        Ok(())
    }

    pub(crate) fn remove<R: RangeBounds<usize>>(&mut self, range: R) -> Vec<Ptr> {
        let result = self.vec.drain(range).collect();
        self.vec.sort();
        result
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }
}

impl<'a> IntoIterator for &'a PtrColumn {
    type Item = &'a Ptr;

    type IntoIter = std::slice::Iter<'a, Ptr>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.iter()
    }
}
impl<'a> IntoIterator for &'a mut PtrColumn {
    type Item = &'a mut Ptr;

    type IntoIter = std::slice::IterMut<'a, Ptr>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.iter_mut()
    }
}

// all with the same id and and must have diff types
#[derive(Clone)]
pub struct PtrRow {
    pub(crate) sparse_index: SparseIndex,
    // not sorted
    pub(crate) data: HashMap<CompType, Ptr>,
}
impl PtrRow {
    pub(crate) fn new_empty(sparse_index: SparseIndex) -> Self {
        Self {
            data: HashMap::new(),
            sparse_index,
        }
    }

    pub(crate) fn push(&mut self, mut access_vec: Vec<Ptr>) -> Result<(), &'static str> {
        // not checking if comp type is being replaced
        for each in access_vec.into_iter() {
            if each.sparse_index != self.sparse_index {
                return Err("sparse index not matching");
            }
            self.data.insert(each.comp_type, each);
        }
        Ok(())
    }

    pub(crate) fn get(&self, comp_type: CompType) -> Result<Ptr, &'static str> {
        Ok(self
            .data
            .get(&comp_type)
            .ok_or("type not present in this row")?
            .clone())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub(crate) struct Generation(usize);
impl Generation {
    pub(crate) fn advance(&mut self) -> Self {
        self.0 += 1;
        *self
    }

    pub(crate) fn clear(&mut self) {
        self.0 = 0;
    }
}

//-----------------SPARSE/DENSE INDEX-----------------//
#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
pub(crate) struct SparseIndex(usize);

#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
pub(crate) struct DenseIndex(usize);

pub(crate) struct SparseVec(Vec<Option<DenseIndex>>);
impl SparseVec {
    pub(crate) fn new(size: usize) -> Self {
        Self(vec![None; size])
    }

    pub(crate) fn resize(&mut self, size: usize) {
        self.0.resize(size, None);
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

impl Index<SparseIndex> for SparseVec {
    type Output = Option<DenseIndex>;

    fn index(&self, index: SparseIndex) -> &Self::Output {
        &self.0[index.0]
    }
}

impl IndexMut<SparseIndex> for SparseVec {
    fn index_mut(&mut self, index: SparseIndex) -> &mut Self::Output {
        &mut self.0[index.0]
    }
}

//----------------SPARSE SET------------------//
pub(crate) struct SparseSet {
    sparse: SparseVec,
    comp_type: CompType,

    capacity: usize,
    len: usize,

    // first is sparse index, second is generation
    data_heap_ptr: *mut u8,
    // in bytes
    generation_offset: usize,
    content_offset: usize,
    unit_layout: Layout,
}

impl SparseSet {
    pub(crate) fn new(comp_type: CompType, size: usize) -> Self {
        let (generation_layout, generation_offset) = Layout::new::<SparseIndex>()
            .extend(Layout::new::<usize>())
            .unwrap();
        let (total_layout, content_offset) = generation_layout.extend(comp_type.layout).unwrap();
        let data_heap_ptr = unsafe { alloc(comp_type.layout.repeat(size).unwrap().0) };
        Self {
            comp_type,
            data_heap_ptr,
            capacity: size,
            len: 0,
            sparse: SparseVec::new(size),
            unit_layout: total_layout,
            generation_offset,
            content_offset,
        }
    }

    //-----------------GET PTR-----------------//
    unsafe fn get_ptr_sparse(&self, dense_index: DenseIndex) -> *mut u8 {
        self.data_heap_ptr
            .add(self.comp_type.layout.size() * dense_index.0)
    }

    unsafe fn get_ptr_gen(&self, dense_index: DenseIndex) -> *mut u8 {
        self.data_heap_ptr
            .add(self.comp_type.layout.size() * dense_index.0 + self.generation_offset)
    }

    unsafe fn get_ptr_content(&self, dense_index: DenseIndex) -> *mut u8 {
        self.data_heap_ptr
            .add(self.comp_type.layout.size() * dense_index.0 + self.content_offset)
    }

    //-----------------GENERATION-----------------//
    unsafe fn gen_read(&self, dense_index: DenseIndex) -> Generation {
        self.get_ptr_gen(dense_index)
            .cast::<Generation>()
            .as_mut()
            .unwrap()
            .clone()
    }

    unsafe fn gen_advance(&self, dense_index: DenseIndex) -> Generation {
        let ptr = self.get_ptr_gen(dense_index);
        ptr.cast::<Generation>().as_mut().unwrap().advance()
    }

    unsafe fn gen_clear(&self, dense_index: DenseIndex) {
        self.get_ptr_gen(dense_index)
            .cast::<Generation>()
            .as_mut()
            .unwrap()
            .clear();
    }

    unsafe fn gen_write(&self, dense_index: DenseIndex, gen: Generation) {
        *self
            .get_ptr_gen(dense_index)
            .cast::<Generation>()
            // todo as ref wouldn't work here
            .as_ref()
            .unwrap() = gen;
    }

    //-----------------SPARSE INDEX-----------------//
    unsafe fn sparse_read(&self, dense_index: DenseIndex) -> SparseIndex {
        self.get_ptr_sparse(dense_index)
            .cast::<SparseIndex>()
            .as_mut()
            .unwrap()
            .clone()
    }

    unsafe fn sparse_write(&self, dense_index: DenseIndex, sparse_index: SparseIndex) {
        *self
            .get_ptr_sparse(dense_index)
            .cast::<SparseIndex>()
            .as_mut()
            .unwrap() = sparse_index;
    }

    //-----------------CONTENT-----------------//
    unsafe fn content_read(&self, dense_index: DenseIndex) -> Ptr {
        let ptr = self.get_ptr_content(dense_index);
        let sparse = self.sparse_read(dense_index);
        Ptr::new(ptr, self.comp_type, sparse)
    }

    unsafe fn content_copy(&self, dense_index: DenseIndex) -> Value {
        self.content_read(dense_index).cast_value()
    }

    // just write content, no touching generation or sparse index
    unsafe fn content_write(&self, dense_index: DenseIndex, ptr: Ptr) {
        let content_ptr = self.get_ptr_content(dense_index);
        copy(ptr.ptr, content_ptr, self.comp_type.layout.size())
    }

    unsafe fn content_replace(&self, dense_index: DenseIndex, ptrs: Ptr) -> Value {
        let result = self.content_copy(dense_index);
        self.content_write(dense_index, ptrs);
        result
    }

    //-----------------ALLOCATION-----------------//
    fn ensure_dense_cap(&mut self, extra_slots: usize) {
        if self.capacity - self.len >= extra_slots {
            self.double_dense_cap();
            self.ensure_dense_cap(extra_slots);
        }
    }

    fn ensure_sparse_cap(&mut self, sparse_index: SparseIndex) {
        let len = self.sparse.0.len();
        if len <= sparse_index.0 {
            self.sparse.0.resize(len * 2, None);
            self.ensure_sparse_cap(sparse_index);
        }
    }

    // todo drop trait
    fn double_dense_cap(&mut self) {
        let new_capacity = self.capacity * 2;
        let (new_layout_of_whole_vec, _) = self
            .unit_layout
            .repeat(new_capacity)
            .expect("could not repeat this layout");
        let new_data_ptr = unsafe {
            realloc(
                self.data_heap_ptr,
                new_layout_of_whole_vec,
                new_layout_of_whole_vec.size(),
            )
        };
        self.capacity = new_capacity;
        self.data_heap_ptr = new_data_ptr;
    }

    //-----------------SPARSE HELPERS-----------------//
    fn dense_read(&self, sparse_index: SparseIndex) -> Result<DenseIndex, &'static str> {
        if sparse_index.0 >= self.sparse.len() {
            Err("index overflow")
        } else {
            if let Some(dense_index) = self.sparse[sparse_index] {
                Ok(dense_index)
            } else {
                Err("attempt to read empty sparse index")
            }
        }
    }

    fn dense_write(
        &mut self,
        sparse_index: SparseIndex,
        dense_index: DenseIndex,
    ) -> Result<(), &'static str> {
        if sparse_index.0 >= self.sparse.len() {
            Err("index overflow")
        } else {
            if let Some(current_index) = self.sparse[sparse_index] {
                self.sparse[sparse_index] = Some(dense_index);
                Ok(())
            } else {
                Err("attempt to write empty sparse index")
            }
        }
    }

    //-----------------DENSE HELPERS-----------------//
    // move last dense element along with its content to the dense index
    unsafe fn tail_move(&self, dense_index: DenseIndex) {}

    // push this content to the tail of dense vec
    unsafe fn push_content(&self, ptr: Ptr) {}

    //-----------------MAIN API-----------------//
    pub(crate) fn is_taken(&self, sparse_index: SparseIndex) -> bool {
        self.dense_read(sparse_index).is_ok()
    }

    // todo start from here
    pub(crate) fn insert(
        &mut self,
        sparse_index: SparseIndex,
        ptrs: Ptr,
    ) -> Result<Ptr, &'static str> {
        if self.dense_read(sparse_index).is_ok() {
            return Err("cell taken");
        }

        if ptrs.comp_type != self.comp_type {
            return Err("wrong type of comp type");
        }

        self.ensure_sparse_cap(sparse_index);
        self.ensure_dense_cap(1);

        let len = DenseIndex(self.len);

        self.sparse[sparse_index] = Some(len);

        unsafe {
            // todo advance gen and write the sparse index
            self.content_write(len, ptrs);
            let raw_dst_ptr = self.content_read(len);
            self.len += 1;
            todo!()
        }
    }

    pub(crate) fn remove(&mut self, sparse_index: SparseIndex) -> Result<Value, &'static str> {
        let dense_index = self.dense_read(sparse_index)?;
        self.sparse[sparse_index] = None;
        let result =
            unsafe { self.replace(dense_index, self.content_read(DenseIndex(self.len - 1)))? };
        self.len -= 1;
        Ok(result)
    }

    // shallow move
    pub(crate) fn move_value(
        &mut self,
        from_index: SparseIndex,
        to_index: SparseIndex,
    ) -> Result<Ptr, &'static str> {
        if self.dense_read(to_index).is_ok() {
            return Err("cell occupied");
        } else {
            let dense_index = self.dense_read(from_index)?;
            self.sparse[to_index] = Some(dense_index);
            let gen = unsafe { self.gen_advance(dense_index) };
            Ok(Ptr::new(to_index, unsafe {
                self.content_read(dense_index)
            }))
        }
    }

    pub(crate) fn replace(
        &mut self,
        sparse_index: SparseIndex,
        ptrs: Ptr,
    ) -> Result<Value, &'static str> {
        unsafe { self.replace(self.dense_read(sparse_index)?, ptrs) }
    }

    // shallow swap of two valid cells
    pub(crate) fn swap_within(
        &mut self,
        sparse_index1: SparseIndex,
        sparse_index2: SparseIndex,
    ) -> Result<(Ptr, Ptr), &'static str> {
        let dense_index1 = self.dense_read(sparse_index1)?;
        let gen1 = unsafe { self.gen_advance(dense_index1) };
        let dense_index2 = self.dense_read(sparse_index2)?;
        let gen2 = unsafe { self.gen_advance(dense_index2) };
        self.sparse[sparse_index1] = Some(dense_index2);
        self.sparse[sparse_index2] = Some(dense_index1);
        Ok((
            Ptr::new(sparse_index1, unsafe { self.content_read(dense_index2) }),
            Ptr::new(sparse_index2, unsafe { self.content_read(dense_index1) }),
        ))
    }

    pub(crate) fn get_cell(&self, sparse_index: SparseIndex) -> Result<Ptr, &'static str> {
        let ptrs = unsafe { self.is_taken(sparse_index)? };
        Ok(Ptr::new(sparse_index, ptrs))
    }

    pub(crate) fn get_column(&self) -> PtrColumn {
        let mut result = PtrColumn::new_empty(self.comp_type);
        for index in 0..self.len {
            let dense_index = DenseIndex(index);
            let ptrs = unsafe { self.content_read(dense_index) };
            result.push(ptrs.to_access());
        }
        result
    }
}

impl Drop for SparseSet {
    fn drop(&mut self) {
        todo!()
    }
}

pub struct Table {
    table: HashMap<CompType, SparseSet>,
    row_type_cache: HashMap<SparseIndex, PtrRow>,
    bottom_sparse_index: SparseIndex,
}

// TODO: incorporate all the query filter methods within the table api, making it a more proper table data structure
// TODO: variadic component insertion, probably with tuple
impl Table {
    pub(crate) fn new() -> Self {
        Self {
            table: HashMap::new(),
            row_type_cache: HashMap::new(),
            bottom_sparse_index: SparseIndex(0),
        }
    }

    //-----------------COLUMN MANIPULATION-----------------//
    pub(crate) fn init_column(&mut self, comp_type: CompType) -> &mut SparseSet {
        if self.try_column(comp_type).is_ok() {
            panic!("type cannot be init twice")
        }
        self.table.insert(comp_type, SparseSet::new(comp_type, 64));
        self.table.get_mut(&comp_type).unwrap()
    }

    pub(crate) fn get_column(&mut self, comp_type: CompType) -> Result<PtrColumn, &'static str> {
        Ok(self.try_column(comp_type)?.get_column())
    }

    pub(crate) fn pop_column(&mut self, comp_type: CompType) -> Option<SparseSet> {
        self.table.remove(&comp_type)
    }

    //-----------------ROW MANIPULATION-----------------//
    pub(crate) fn init_row(&mut self) -> SparseIndex {
        let result = self.bottom_sparse_index;
        self.row_type_cache.insert(
            self.bottom_sparse_index,
            PtrRow::new_empty(self.bottom_sparse_index),
        );
        self.bottom_sparse_index.0 += 1;
        result
    }

    pub(crate) fn get_row(&mut self, sparse_index: SparseIndex) -> Result<PtrRow, &'static str> {
        // since init row ensures the existence of all row cache
        let cache = self
            .row_type_cache
            .get(&sparse_index)
            .ok_or("index overflow")?;
        Ok(cache.clone())
    }

    //-----------------HELPERS-----------------//
    fn try_column(&mut self, comp_type: CompType) -> Result<&mut SparseSet, &'static str> {
        if let Some(access) = self.table.get_mut(&comp_type) {
            Ok(access)
        } else {
            Err("no such type/column")
        }
    }

    fn ensure_column(&mut self, comp_type: CompType) -> &mut SparseSet {
        self.table
            .entry(comp_type)
            .or_insert(SparseSet::new(comp_type, 64))
    }

    // bypassing generation
    pub(crate) fn try_cell(
        &mut self,
        comp_type: CompType,
        sparse_index: SparseIndex,
    ) -> Result<Ptr, &'static str> {
        unsafe { self.try_column(comp_type)?.is_taken(sparse_index) }
    }

    //-----------------CELL IO OPERATION-----------------//

    // if not column init, init it automatically
    pub(crate) fn push_cell(
        &mut self,
        sparse_index: SparseIndex,
        // todo: maybe Value??
        values: Ptr,
    ) -> Result<Ptr, &'static str> {
        let result = self
            .ensure_column(values.comp_type)
            .insert(values, sparse_index)?;
        self.row_type_cache
            .get_mut(&sparse_index)
            .ok_or("invalid row")?
            .push(result)?;
        Ok(result)
    }

    pub(crate) fn pop_cell(&mut self, access: Ptr) -> Result<Value, &'static str> {
        self.row_type_cache
            .get_mut(&access.sparse_index)
            .ok_or("invalid row")?
            .pop(access.ptr.comp_type)?;
        self.try_column(access.ptr.comp_type)?
            .remove(access.sparse_index)
    }

    pub(crate) fn replace_cell(
        &mut self,
        access: Ptr,
        // todo maybe value?
        values: Ptr,
    ) -> Result<Value, &'static str> {
        self.row_type_cache
            .get_mut(&access.sparse_index)
            .ok_or("invalid row")?
            .pop(access.ptr.comp_type)?;
        self.row_type_cache
            .get_mut(&access.sparse_index)
            .ok_or("invalid row")?
            .push(values.to_access())?;
        self.try_column(access.ptr.comp_type)?
            .replace(access.sparse_index, values)
    }

    //-----------------CELL OPERATION WITHIN TABLE-----------------//

    /// one valid cell move to an empty one, returns the new table cell access
    pub(crate) fn move_cell_within(
        &mut self,
        from_key: Ptr,
        to_index: SparseIndex,
    ) -> Result<Ptr, &'static str> {
        if self.try_cell(from_key.ptr.comp_type, to_index).is_ok() {
            return Err("cell not vacant");
        } else {
            let access = self
                .row_type_cache
                .get_mut(&from_key.sparse_index)
                .ok_or("invalid row")?
                .pop(from_key.ptr.comp_type)?;
            self.row_type_cache
                .get_mut(&to_index)
                .ok_or("invalid row")?
                .push(access)?;
            let result = self
                .try_column(from_key.ptr.comp_type)?
                .insert(Ptr::new(from_key.ptr.ptr, from_key.ptr.comp_type), to_index)?;
            self.pop_cell(from_key)?;
            Ok(result)
        }
    }

    /// two valid cells, move one to another location, and pop that location
    pub(crate) fn replace_cell_within(
        &mut self,
        from_key: Ptr,
        to_key: Ptr,
    ) -> Result<(Value, Ptr), &'static str> {
        if from_key.column_type() != to_key.column_type() {
            return Err("not on the same column");
        }
        let cached_access = self
            .row_type_cache
            .get_mut(&from_key.sparse_index)
            .ok_or("invalid row")?
            .pop(to_key.ptr.comp_type)?;
        self.row_type_cache
            .get_mut(&to_key.sparse_index)
            .ok_or("invalid row")?
            .push(cached_access)?;

        let result = self
            .try_column(from_key.ptr.comp_type)?
            .remove(to_key.sparse_index)?;
        let access = self
            .try_column(from_key.ptr.comp_type)?
            .move_value(from_key.sparse_index, to_key.sparse_index)?;
        Ok((result, access))
    }

    /// shallow swap between two valid cells
    pub(crate) fn swap_cell_within(
        &mut self,
        key1: Ptr,
        key2: Ptr,
    ) -> Result<(Ptr, Ptr), &'static str> {
        if key1.column_type() != key2.column_type() {
            return Err("not on the same column");
        }
        let cached_access1 = self
            .row_type_cache
            .get_mut(&key1.sparse_index)
            .ok_or("invalid row")?
            .pop(key1.ptr.comp_type)?;
        let cached_access2 = self
            .row_type_cache
            .get_mut(&key2.sparse_index)
            .ok_or("invalid row")?
            .pop(key1.ptr.comp_type)?;
        self.row_type_cache
            .get_mut(&key1.sparse_index)
            .ok_or("invalid row")?
            .push(cached_access2)?;
        self.row_type_cache
            .get_mut(&key2.sparse_index)
            .ok_or("invalid row")?
            .push(cached_access1)?;

        self.try_column(key1.column_type())?
            .swap_within(key1.sparse_index(), key2.sparse_index())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ExecutionFrequency {
    Always,
    Once(bool),
    // Timed(f64, f64),
}

pub struct With<FilterComp: Component>(pub(crate) PhantomData<FilterComp>);
impl<FilterComp: Component> With<FilterComp> {
    // all these access would have the same type but different id
    pub(crate) fn apply_with_filter(mut vec: PtrColumn, table: &mut Table) -> PtrColumn {
        todo!()
    }
}

pub struct Without<FilterComp: Component>(pub(crate) PhantomData<FilterComp>);
impl<FilterComp: Component> Without<FilterComp> {
    pub(crate) fn apply_without_filter(mut vec: PtrColumn, table: &mut Table) -> PtrColumn {
        todo!()
    }
}

pub trait Filter: Sized {
    fn apply_on(vec: PtrColumn, table: &mut Table) -> PtrColumn;
}
impl<FilterComp: Component> Filter for With<FilterComp> {
    fn apply_on(vec: PtrColumn, table: &mut Table) -> PtrColumn {
        With::<FilterComp>::apply_with_filter(vec, table)
    }
}
impl<FilterComp: Component> Filter for Without<FilterComp> {
    fn apply_on(vec: PtrColumn, table: &mut Table) -> PtrColumn {
        Without::<FilterComp>::apply_without_filter(vec, table)
    }
}
impl Filter for () {
    fn apply_on(vec: PtrColumn, table: &mut Table) -> PtrColumn {
        vec
    }
}

pub struct Command<'a> {
    table: &'a mut Table,
}

// TODO: turns the api into wrapper functions of those in impl ComponentTable
impl<'a> Command<'a> {
    pub(crate) fn new(table: &'a mut Table) -> Self {
        Self { table }
    }

    pub fn add_component<C: Component>(&mut self, mut component: C) -> Ptr {
        todo!()
    }

    // key or entity index? usize or generational index?
    pub fn attach_component<C: Component>(
        &mut self,
        key: Ptr,
        mut component: C,
    ) -> Result<Ptr, &'static str> {
        todo!()
    }

    pub fn remove_component<C: Component>(&mut self, key: Ptr) -> Result<C, &'static str> {
        todo!()
    }

    pub fn query<C: Component, F: Filter>(&mut self) -> PtrColumn {
        todo!()
    }
}

pub struct System {
    pub(crate) order: usize,
    pub(crate) frequency: ExecutionFrequency,
    pub(crate) func: fn(Command),
}
impl System {
    pub fn default(func: fn(Command)) -> Self {
        Self {
            order: 0,
            frequency: ExecutionFrequency::Always,
            func,
        }
    }

    pub fn new(order: usize, frequency: ExecutionFrequency, func: fn(Command)) -> Self {
        Self {
            order,
            frequency,
            func,
        }
    }

    pub(crate) fn run(&self, table: &mut Table) {
        (self.func)(Command::new(table))
    }

    fn is_once_run(&self) -> bool {
        match self.frequency {
            ExecutionFrequency::Always => false,
            ExecutionFrequency::Once(run_status) => run_status,
            // _ => non exhaustive,
        }
    }
}

impl PartialEq for System {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}
impl Eq for System {}
impl PartialOrd for System {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.order.partial_cmp(&other.order)
    }
}
impl Ord for System {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.order.cmp(&other.order)
    }
}

pub struct Scheduler {
    new_pool: Vec<System>,
    // waiting: Vec<System>,
    pub(crate) queue: Vec<System>,
}
impl Scheduler {
    pub fn new() -> Self {
        Self {
            new_pool: vec![],
            // waiting: vec![],
            queue: vec![],
        }
    }

    pub fn add_system(&mut self, system: System) {
        self.new_pool.push(system);
    }

    pub(crate) fn prepare_queue(&mut self) {
        self.queue.retain(|x| !x.is_once_run());
        if !self.new_pool.is_empty() {
            self.queue.append(&mut self.new_pool);
            self.new_pool.clear();
        }
        self.queue.sort();
    }
}

pub struct ECS {
    table: Table,
    scheduler: Scheduler,
}

impl ECS {
    pub fn new() -> Self {
        Self {
            table: Table::new(),
            scheduler: Scheduler::new(),
        }
    }

    pub fn add_system(&mut self, func: fn(Command), order: usize, once: bool) {
        match once {
            true => self.scheduler.add_system(System {
                order,
                frequency: ExecutionFrequency::Once(false),
                func,
            }),
            false => self.scheduler.add_system(System {
                order,
                frequency: ExecutionFrequency::Always,
                func,
            }),
        }
    }

    pub fn tick(&mut self) {
        self.scheduler.prepare_queue();
        for system in &mut self.scheduler.queue {
            match system.frequency {
                ExecutionFrequency::Always => system.run(&mut self.table),
                ExecutionFrequency::Once(_) => {
                    system.frequency = ExecutionFrequency::Once(true);
                    system.run(&mut self.table);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[derive(Clone)]
    struct Health(i32);
    impl Component for Health {}

    #[derive(Clone, Debug)]
    struct Mana(i32);
    impl Component for Mana {}

    #[derive(Clone, Debug)]
    struct Player(&'static str);
    impl Component for Player {}

    fn spawn(mut command: Command) {
        let key = command.add_component(Player("test player uwu"));
    }

    fn say_hi(mut command: Command) {}

    fn remove(mut command: Command) {
        // for pl in &mut command.query::<Player, ()>() {
        //     command.remove_component::<Player>(*pl).unwrap();
        // }
        todo!()
    }

    #[test]
    fn test() {
        let mut ecs = ECS::new();
        ecs.add_system(spawn, 1, true);
        ecs.add_system(say_hi, 2, false);

        ecs.tick();
        ecs.tick();
        ecs.tick();
        ecs.tick();
        ecs.tick();
        ecs.tick();
        ecs.add_system(remove, 0, true);
        ecs.tick();
        ecs.tick();
        ecs.tick();
        ecs.tick();
    }
}
