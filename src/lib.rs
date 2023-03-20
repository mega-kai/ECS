#![allow(dead_code, unused_variables, unused_imports, unused_mut)]
#![feature(alloc_layout_extra, map_try_insert, core_intrinsics, const_trait_impl)]
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::{Add, Index, IndexMut, Range, RangeBounds};
use std::slice::SliceIndex;
use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    collections::HashMap,
    fmt::Debug,
};

const GENERATION_COMPTYPE: CompType = CompType::new::<Generation>();
const SPARSE_INDEX_COMPTYPE: CompType = CompType::new::<SparseIndex>();

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableCellAccess {
    // pub(crate) assigned_index: usize,
    pub(crate) sparse_index: SparseIndex,
    pub(crate) column_type: MultiCompType,
    pub(crate) ptr: *mut u8,
    pub(crate) generation: Generation,
}
impl TableCellAccess {
    pub(crate) fn new(
        sparse_index: SparseIndex,
        column_type: MultiCompType,
        ptr: *mut u8,
        generation: Generation,
    ) -> Self {
        Self {
            sparse_index,
            column_type,
            ptr,
            generation,
        }
    }
    pub(crate) fn sparse_index(&self) -> SparseIndex {
        self.sparse_index
    }
    pub(crate) fn column_type(&self) -> MultiCompType {
        self.column_type
    }
}

// all with the same component type
#[derive(Clone)]
pub struct AccessColumn(pub(crate) Vec<TableCellAccess>, pub(crate) CompType);
impl AccessColumn {
    pub(crate) fn new_empty(comp_type: CompType) -> Self {
        Self(vec![], comp_type)
    }

    pub fn cast_vec<C: Component>(&self) -> Vec<&mut C> {
        // on the promise that all accesses of this vec share the same type
        // assert_eq!(C::id(), self.0[0].column_index);
        self.into_iter()
            .map(|x| unsafe { x.ptr.cast::<C>().as_mut().unwrap() })
            .collect::<Vec<&mut C>>()
    }
}
impl<'a> IntoIterator for &'a AccessColumn {
    type Item = &'a TableCellAccess;

    type IntoIter = std::slice::Iter<'a, TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl<'a> IntoIterator for &'a mut AccessColumn {
    type Item = &'a mut TableCellAccess;

    type IntoIter = std::slice::IterMut<'a, TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

// all with the same id and and must have diff types
#[derive(Clone)]
pub struct AccessRow {
    // not ordered; CONSIDER: turn it into a hash map where k:comp_type
    pub(crate) access_vec: Vec<TableCellAccess>,
    pub(crate) sparse_index: SparseIndex,
}
impl AccessRow {
    pub(crate) fn new(access_vec: Vec<TableCellAccess>, sparse_index: SparseIndex) -> Self {
        Self {
            access_vec,
            sparse_index,
        }
    }

    pub(crate) fn new_empty(sparse_index: SparseIndex) -> Self {
        Self {
            access_vec: vec![],
            sparse_index,
        }
    }

    pub(crate) fn get_access_from_type(
        &self,
        comp_type: MultiCompType,
    ) -> Result<TableCellAccess, &'static str> {
        let mut counter: usize = 0;
        let mut final_index: usize = 0;
        for (index, access) in self.into_iter().enumerate() {
            if access.column_type() == comp_type {
                counter += 1;
                final_index = index;
            }
        }
        match counter {
            0 => Err("zero of this type in this row"),
            1 => Ok(self.access_vec[final_index].clone()),
            _ => Err("more than one of this type in this row"),
        }
    }

    pub(crate) fn contains_access(&self, key: TableCellAccess) -> bool {
        self.access_vec.contains(&key)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.access_vec.is_empty()
    }

    pub(crate) fn get_current_generation(
        &self,
        comp_type: MultiCompType,
    ) -> Result<Generation, &'static str> {
        Ok(self.get_access_from_type(comp_type)?.generation)
    }
}
impl<'a> IntoIterator for &'a AccessRow {
    type Item = &'a TableCellAccess;

    type IntoIter = std::slice::Iter<'a, TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.access_vec.iter()
    }
}
impl<'a> IntoIterator for &'a mut AccessRow {
    type Item = &'a mut TableCellAccess;

    type IntoIter = std::slice::IterMut<'a, TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.access_vec.iter_mut()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompType {
    pub(crate) type_id: TypeId,
    pub(crate) layout: Layout,
}
impl CompType {
    pub(crate) fn new<C: 'static>() -> Self {
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

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub(crate) struct Generation(usize);
impl Generation {
    pub(crate) fn advance(&mut self) -> Self {
        self.0 += 1;
        self.clone()
    }

    pub(crate) fn clear(&mut self) {
        self.0 = 0;
    }
}

/// built in sparse index and generation: sparse index offset would always be 0, followed by generation
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MultiCompType {
    hashmap_layout_and_offsets: HashMap<CompType, (Layout, usize)>,
    total_layout: Layout,
}
impl MultiCompType {
    pub(crate) fn new_empty() -> Self {
        let mut layout_and_offsets = HashMap::new();
        let mut total_layout = Layout::new::<SparseIndex>();
        let (with_generation_layout, generation_offset) =
            total_layout.extend(Layout::new::<Generation>()).unwrap();
        total_layout = with_generation_layout;
        layout_and_offsets.insert(
            CompType::new::<SparseIndex>(),
            (Layout::new::<SparseIndex>(), 0),
        );
        layout_and_offsets.insert(
            CompType::new::<Generation>(),
            (Layout::new::<Generation>(), generation_offset),
        );

        Self {
            hashmap_layout_and_offsets: layout_and_offsets,
            total_layout,
        }
    }

    pub(crate) fn add(&mut self, comp_types: Vec<CompType>) -> Result<(), &'static str> {
        if comp_types.is_empty() {
            return Ok(());
        }
        for ty in comp_types.iter() {
            if ty.layout.size() == 0 {
                return Err("zst");
            }
            if self.get_layout_and_offset(*ty).is_ok() {
                return Err("repeated type");
            }
        }

        let final_layout = self.total_layout;

        let (with_generation_layout, generation_offset) = final_layout
            .extend(Layout::new::<Generation>())
            .or(Err("cannot extend"))?;
        for ty in comp_types {
            let (layout, offset) = final_layout.extend(ty.layout).or(Err("cannot extend"))?;
            self.hashmap_layout_and_offsets.insert(ty, (layout, offset));
        }
        self.total_layout = final_layout;

        Ok(())
    }

    pub(crate) fn new(mut comp_types: Vec<CompType>) -> Result<Self, &'static str> {
        let mut val = Self::new_empty();
        val.add(comp_types)?;
        Ok(val)
    }

    pub(crate) fn total_layout(&self) -> Layout {
        self.total_layout
    }

    pub(crate) fn get_layout_and_offset(
        &self,
        comp_type: CompType,
    ) -> Result<(Layout, usize), &'static str> {
        let result = self
            .hashmap_layout_and_offsets
            .get(&comp_type)
            .ok_or("type not in this comp types")?
            .clone();
        Ok(result)
    }

    pub(crate) fn len(&self) -> usize {
        self.hashmap_layout_and_offsets.len()
    }
}

impl Hash for MultiCompType {
    fn hash_slice<H: ~const std::hash::Hasher>(data: &[Self], state: &mut H)
    where
        Self: Sized,
    {
        //FIXME(const_trait_impl): revert to only a for loop
        fn rt<T: Hash, H: std::hash::Hasher>(data: &[T], state: &mut H) {
            for piece in data {
                piece.hash(state)
            }
        }
        const fn ct<T: ~const Hash, H: ~const std::hash::Hasher>(data: &[T], state: &mut H) {
            let mut i = 0;
            while i < data.len() {
                data[i].hash(state);
                i += 1;
            }
        }
        // SAFETY: same behavior, CT just uses while instead of for
        unsafe { std::intrinsics::const_eval_select((data, state), ct, rt) };
    }

    fn hash<H: ~const std::hash::Hasher>(&self, state: &mut H) {
        self.total_layout.hash(state);
    }
}

//-----------------SINGLE TYPE POINTERS-----------------//
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Ptr {
    pub(crate) ptr: *mut u8,
    pub(crate) comp_type: CompType,
}
impl Ptr {
    pub(crate) fn new(ptr: *mut u8, comp_type: CompType) -> Self {
        Self { ptr, comp_type }
    }

    pub(crate) fn cast_value(self) -> Value {
        let mut result = Value::new_empty(self.comp_type);
        unsafe {
            result.write(self.ptr);
        }
        result
    }

    pub(crate) unsafe fn cast<T>(&self) -> Result<&mut T, &'static str> {
        if CompType::new::<T>() != self.comp_type {
            return Err("type not matching");
        } else {
            Ok(self.ptr.cast::<T>().as_mut().ok_or("casting failure")?)
        }
    }
}

pub(crate) struct MultiPtr {
    ptr: *mut u8,
    comp_type: MultiCompType,
}
impl MultiPtr {
    pub(crate) fn new(ptr: *mut u8, comp_type: MultiCompType) -> Self {
        Self { ptr, comp_type }
    }

    pub(crate) fn cast_multi_value(self) -> MultiValue {
        let mut result = MultiValue::new_empty(self.comp_type);
        unsafe {
            result.write(self.ptr);
        }
        result
    }

    pub(crate) fn get_sparse_index(&self) -> Option<usize> {
        unsafe { self.ptr.cast::<Option<usize>>().as_ref().unwrap().clone() }
    }

    pub(crate) fn get_gen(&self) -> Generation {
        unsafe {
            self.ptr
                .add(
                    self.comp_type
                        .get_layout_and_offset(GENERATION_COMPTYPE)
                        .unwrap()
                        .1,
                )
                .cast::<Generation>()
                .as_ref()
                .unwrap()
                .clone()
        }
    }
}
//-----------------TYPE ERASED VALUES-----------------//
// todo refactor this into a allocated ptr on heap with drop trait deallocating it
pub(crate) struct Value {
    pub(crate) val: Vec<u8>,
    pub(crate) comp_type: CompType,
}
impl Value {
    pub(crate) fn new_empty(comp_type: CompType) -> Self {
        Self {
            val: vec![0u8; comp_type.layout.size()],
            comp_type,
        }
    }

    pub(crate) unsafe fn write(&mut self, ptr: *mut u8) {
        unsafe { std::ptr::copy(ptr, self.val.as_mut_ptr(), self.comp_type.layout.size()) }
    }

    pub(crate) unsafe fn cast<T: Clone>(self) -> Result<T, &'static str> {
        if self.comp_type != CompType::new::<T>() {
            return Err("type not matching");
        } else {
            Ok(self.val.as_ptr().cast::<T>().as_ref().unwrap().clone())
        }
    }
}

pub(crate) struct MultiValue {
    vec: Vec<u8>,
    pub(crate) comp_type: MultiCompType,
}
impl MultiValue {
    pub(crate) fn new_empty(comp_type: MultiCompType) -> Self {
        Self {
            vec: vec![0u8; comp_type.total_layout.size()],
            comp_type,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.vec.len()
    }

    pub(crate) unsafe fn write(&mut self, ptr: *mut u8) {
        unsafe {
            std::ptr::copy(
                ptr,
                self.vec.as_mut_ptr(),
                self.comp_type.total_layout.size(),
            )
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub(crate) struct DenseIndex(usize);

#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
pub(crate) struct SparseIndex(usize);

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
    ty: MultiCompType,
    data_heap_ptr: *mut u8,
    pub(crate) capacity: usize,
    pub(crate) len: usize,
    pub(crate) sparse: SparseVec,
}

// todo: make operations unsafe as they should be
// todo: turn all sparse_index into SparseIndex
impl SparseSet {
    pub(crate) fn new(comp_types: MultiCompType, size: usize) -> Self {
        let data_heap_ptr =
            unsafe { std::alloc::alloc(comp_types.total_layout().repeat(size).unwrap().0) };
        Self {
            ty: comp_types,
            data_heap_ptr,
            capacity: size,
            len: 0,
            sparse: SparseVec::new(size),
        }
    }

    //-----------------HELPERS-----------------//
    /// must ensure dense_index is valid first
    fn get_single_dense_ptr(
        &self,
        dense_index: DenseIndex,
        comp_type: CompType,
    ) -> Result<*mut u8, &'static str> {
        let (_, offset) = self.ty.get_layout_and_offset(comp_type)?;
        unsafe {
            Ok(self
                .data_heap_ptr
                .add(self.ty.total_layout().size() * dense_index.0 + offset))
        }
    }

    fn get_dense_index(&self, sparse_index: SparseIndex) -> Result<DenseIndex, &'static str> {
        if sparse_index.0 >= self.sparse.len() {
            Err("index overflow")
        } else {
            if let Some(dense_index) = self.sparse[sparse_index] {
                Ok(dense_index)
            } else {
                Err("empty sparse index/no such row")
            }
        }
    }

    fn get_sparse_index(&self, dense_index: usize) -> usize {
        unsafe {
            self.data_heap_ptr
                .add(self.ty.total_layout().size() * dense_index)
                .cast::<usize>()
                .as_mut()
                .unwrap()
                .clone()
        }
    }

    pub(crate) unsafe fn read_raw(
        &self,
        sparse_index: SparseIndex,
    ) -> Result<MultiPtr, &'static str> {
        let dense_index = self.get_dense_index(sparse_index)?;
        Ok(self.read_multi(dense_index))
    }

    //-----------------GENERATION OPERATIONS-----------------//
    pub(crate) fn get_gen_val_sparse(
        &self,
        sparse_index: SparseIndex,
    ) -> Result<Generation, &'static str> {
        let dense_index = self.get_dense_index(sparse_index)?;
        unsafe {
            self.copy_single(dense_index, GENERATION_COMPTYPE)?
                .cast::<Generation>()
        }
    }

    pub(crate) unsafe fn get_gen_dense(&self, dense_index: DenseIndex) -> &mut Generation {
        let ptr = self
            .read_single(dense_index, CompType::new::<Generation>())
            .unwrap();
        ptr.cast::<Generation>().unwrap()
    }

    pub(crate) unsafe fn generation_advance(&self, dense_index: DenseIndex) -> Generation {
        let ptr = self.get_gen_dense(dense_index);
        ptr.advance()
    }

    pub(crate) unsafe fn generation_clear(&self, dense_index: DenseIndex) {
        self.get_gen_dense(dense_index).clear();
    }

    pub(crate) unsafe fn generation_write(&self, dense_index: DenseIndex, gen: Generation) {
        *self.get_gen_dense(dense_index) = gen;
    }

    //-----------------DENSE OPERATIONS-----------------//
    pub(crate) unsafe fn read_single(
        &self,
        dense_index: DenseIndex,
        comp_type: CompType,
    ) -> Result<Ptr, &'static str> {
        if comp_type == GENERATION_COMPTYPE || comp_type == SPARSE_INDEX_COMPTYPE {
            return Err("trying to access gen/sparse index");
        }
        Ok(Ptr::new(
            self.get_single_dense_ptr(dense_index, comp_type)?,
            comp_type,
        ))
    }

    pub(crate) unsafe fn read_multi(&self, dense_index: DenseIndex) -> MultiPtr {
        let ptr = self
            .data_heap_ptr
            .add(dense_index.0 * self.ty.total_layout.size());
        MultiPtr::new(ptr, self.ty)
    }

    pub(crate) unsafe fn write_single(
        &self,
        dense_index: DenseIndex,
        owning_ptr: Ptr,
    ) -> Result<Generation, &'static str> {
        if owning_ptr.comp_type == GENERATION_COMPTYPE
            || owning_ptr.comp_type == SPARSE_INDEX_COMPTYPE
        {
            return Err("trying to access gen/sparse index");
        }
        let dst_ptr = self.data_heap_ptr.add(
            dense_index.0 * self.ty.total_layout().size()
                + self.ty.get_layout_and_offset(owning_ptr.comp_type)?.1,
        );
        std::ptr::copy(owning_ptr.ptr, dst_ptr, owning_ptr.comp_type.layout.size());
        Ok(self.generation_advance(dense_index))
    }

    pub(crate) unsafe fn write_multi(
        &self,
        dense_index: DenseIndex,
        owning_ptr: MultiPtr,
    ) -> Generation {
        let mut previous_gen = self.get_gen_dense(dense_index).clone();
        let dst_ptr = self
            .data_heap_ptr
            .add(dense_index.0 * self.ty.total_layout().size());
        std::ptr::copy(owning_ptr.ptr, dst_ptr, self.ty.total_layout.size());
        let result_gen = previous_gen.advance();
        self.generation_write(dense_index, result_gen);
        result_gen
    }

    pub(crate) unsafe fn copy_single(
        &self,
        dense_index: DenseIndex,
        comp_type: CompType,
    ) -> Result<Value, &'static str> {
        Ok(self.read_single(dense_index, comp_type)?.cast_value())
    }

    pub(crate) unsafe fn copy_multi(&self, dense_index: DenseIndex) -> MultiValue {
        self.read_multi(dense_index).cast_multi_value()
    }

    pub(crate) unsafe fn replace_single(
        &self,
        dense_index: DenseIndex,
        src_ptr: Ptr,
    ) -> Result<Value, &'static str> {
        let result = self.copy_single(dense_index, src_ptr.comp_type)?;
        self.write_single(dense_index, src_ptr)?;
        Ok(result)
    }

    pub(crate) unsafe fn replace_multi(
        &self,
        ptrs: MultiPtr,
        dense_index: DenseIndex,
    ) -> Result<MultiValue, &'static str> {
        let result = self.copy_multi(dense_index);
        self.write_multi(dense_index, ptrs);
        Ok(result)
    }

    //-----------------SPARSE OPERATIONS-----------------//

    pub(crate) fn ensure_len(&mut self, sparse_index: SparseIndex) {
        if sparse_index.0 >= self.sparse.len() {
            self.sparse.resize(sparse_index.0 + 1);
        }
        if self.len >= self.capacity {
            self.double_dense_cap();
        }
    }

    pub(crate) fn try_insert(
        &mut self,
        ptrs: MultiPtr,
        sparse_index: SparseIndex,
    ) -> Result<TableCellAccess, &'static str> {
        if self.get_dense_index(sparse_index).is_ok() {
            return Err("cell taken");
        }

        if ptrs.comp_type != self.ty {
            return Err("wrong type of multi comp type");
        }

        self.ensure_len(sparse_index);

        let len = DenseIndex(self.len);

        self.sparse[sparse_index] = Some(len);

        unsafe {
            let gen = self.write_multi(len, ptrs);
            let raw_dst_ptr = self.read_multi(len);
            self.len += 1;
            Ok(TableCellAccess::new(
                sparse_index,
                self.ty,
                raw_dst_ptr.ptr,
                gen,
            ))
        }
    }

    pub(crate) fn remove(&mut self, sparse_index: SparseIndex) -> Result<MultiValue, &'static str> {
        let dense_index = self.get_dense_index(sparse_index)?;
        self.sparse[sparse_index] = None;
        let mut result = MultiValue::new_empty(self.ty);
        let val = unsafe { self.read_multi(dense_index).cast_multi_value() };
        Ok(val)
    }

    pub(crate) fn move_value(
        &mut self,
        from_index: SparseIndex,
        to_index: SparseIndex,
    ) -> Result<TableCellAccess, &'static str> {
        if self.get_dense_index(to_index).is_ok() {
            return Err("cell occupied");
        } else {
            let dense_index = self.get_dense_index(from_index)?;
            self.sparse[to_index] = Some(dense_index);
            let gen = unsafe { self.generation_advance(dense_index) };
            Ok(TableCellAccess::new(
                to_index,
                self.ty,
                unsafe { self.read_multi(dense_index).ptr },
                gen,
            ))
        }
    }

    pub(crate) fn replace(
        &mut self,
        sparse_index: SparseIndex,
        ptrs: MultiPtr,
    ) -> Result<MultiValue, &'static str> {
        unsafe { self.replace_multi(ptrs, self.get_dense_index(sparse_index)?) }
    }

    // "shallow swap" of two valid cells
    // todo deal with the generation of swapping
    pub(crate) fn swap_within(
        &mut self,
        sparse_index1: SparseIndex,
        sparse_index2: SparseIndex,
    ) -> Result<(TableCellAccess, TableCellAccess), &'static str> {
        let dense_index1 = self.get_dense_index(sparse_index1)?;
        let gen1 = unsafe { self.generation_advance(dense_index1) };
        let dense_index2 = self.get_dense_index(sparse_index2)?;
        let gen2 = unsafe { self.generation_advance(dense_index2) };
        self.sparse[sparse_index1] = Some(dense_index2);
        self.sparse[sparse_index2] = Some(dense_index1);
        Ok((
            TableCellAccess::new(
                sparse_index1,
                self.ty,
                unsafe { self.read_multi(dense_index2).ptr },
                gen1,
            ),
            TableCellAccess::new(
                sparse_index2,
                self.ty,
                unsafe { self.read_multi(dense_index1).ptr },
                gen2,
            ),
        ))
    }

    pub(crate) fn get_cell(
        &self,
        sparse_index: SparseIndex,
    ) -> Result<TableCellAccess, &'static str> {
        todo!()
    }

    pub(crate) fn get_column(&self) -> AccessColumn {
        todo!()
    }

    //-----------------ALLOCATION-----------------//
    fn double_dense_cap(&mut self) {
        let new_capacity = self.capacity * 2;
        let (new_layout_of_whole_vec, _) = self
            .ty
            .total_layout()
            .repeat(new_capacity)
            .expect("could not repeat this layout");
        let new_data_ptr = unsafe {
            std::alloc::realloc(
                self.data_heap_ptr,
                self.ty
                    .total_layout()
                    .repeat(self.capacity)
                    .expect("could not repeat layout")
                    .0,
                new_layout_of_whole_vec.size(),
            )
        };
        self.capacity = new_capacity;
        self.data_heap_ptr = new_data_ptr;
    }
}

pub struct MultiTable {
    table: HashMap<MultiCompType, SparseSet>,
    row_type_cache: HashMap<SparseIndex, AccessRow>,
    bottom_sparse_index: SparseIndex,
}

// TODO: cache all the comp types of all the rows, update the cache upon add/attach/remove/swap
// TODO: refactoring api to location/pointer/generation seperate
// TODO: incorporate all the query filter methods within the table api, making it a more proper table data structure
// TODO: variadic component insertion, probably with tuple
impl MultiTable {
    pub(crate) fn new() -> Self {
        Self {
            table: HashMap::new(),
            row_type_cache: HashMap::new(),
            bottom_sparse_index: SparseIndex(0),
        }
    }

    //-----------------COLUMN MANIPULATION-----------------//
    pub(crate) fn init_column(&mut self, comp_type: MultiCompType) -> &mut SparseSet {
        if self.try_column(comp_type).is_ok() {
            panic!("type cannot be init twice")
        }
        self.table.insert(comp_type, SparseSet::new(comp_type, 64));
        self.table.get_mut(&comp_type).unwrap()
    }

    pub(crate) fn get_column(
        &mut self,
        comp_type: MultiCompType,
    ) -> Result<AccessColumn, &'static str> {
        Ok(self.try_column(comp_type)?.get_column())
    }

    pub(crate) fn pop_column(&mut self, comp_type: MultiCompType) -> Option<SparseSet> {
        self.table.remove(&comp_type)
    }

    //-----------------ROW MANIPULATION-----------------//
    pub(crate) fn init_row(&mut self) -> SparseIndex {
        let result = self.bottom_sparse_index;
        self.row_type_cache.insert(
            self.bottom_sparse_index,
            AccessRow::new_empty(self.bottom_sparse_index),
        );
        self.bottom_sparse_index.0 += 1;
        result
    }

    pub(crate) fn get_row(&mut self, sparse_index: SparseIndex) -> Result<AccessRow, &'static str> {
        // since init row ensures the existence of all row cache
        let cache = self
            .row_type_cache
            .get(&sparse_index)
            .ok_or("index overflow")?;
        Ok(cache.clone())
    }

    //-----------------HELPERS-----------------//
    fn try_column(&mut self, comp_type: MultiCompType) -> Result<&mut SparseSet, &'static str> {
        if let Some(access) = self.table.get_mut(&comp_type) {
            Ok(access)
        } else {
            Err("no such type/column")
        }
    }

    fn ensure_column(&mut self, comp_type: MultiCompType) -> &mut SparseSet {
        self.table
            .entry(comp_type)
            .or_insert(SparseSet::new(comp_type, 64))
    }

    // bypassing generation
    pub(crate) fn try_cell(
        &mut self,
        comp_type: MultiCompType,
        sparse_index: SparseIndex,
    ) -> Result<MultiPtr, &'static str> {
        unsafe { self.try_column(comp_type)?.read_raw(sparse_index) }
    }

    //-----------------CELL IO OPERATION-----------------//

    // if not column init, init it automatically
    pub(crate) fn push_cell(
        &mut self,
        sparse_index: SparseIndex,
        values: MultiPtr,
    ) -> Result<TableCellAccess, &'static str> {
        // todo cache
        self.ensure_column(values.comp_type)
            .try_insert(values, sparse_index)
    }

    pub(crate) fn pop_cell(&mut self, access: TableCellAccess) -> Result<MultiValue, &'static str> {
        // todo cache
        self.try_column(access.column_type)?
            .remove(access.sparse_index)
    }

    /// write and return the old one in a series of bytes in a vector
    /// it is on the caller to ensure they are the same type, else it's UB
    pub(crate) fn replace_cell(
        &mut self,
        key: TableCellAccess,
        values: MultiPtr,
    ) -> Result<MultiValue, &'static str> {
        // todo cache
        self.try_column(key.column_type)?
            .replace(key.sparse_index, values)
    }

    //-----------------CELL OPERATION WITHIN TABLE-----------------//

    /// one valid cell move to an empty one, returns the new table cell access
    pub(crate) fn move_cell_within(
        &mut self,
        from_key: TableCellAccess,
        to_index: SparseIndex,
    ) -> Result<TableCellAccess, &'static str> {
        if self.try_cell(from_key.column_type, to_index).is_ok() {
            return Err("cell not vacant");
        } else {
            let result = self
                .try_column(from_key.column_type)?
                .try_insert(MultiPtr::new(from_key.ptr, from_key.column_type), to_index)?;
            self.pop_cell(from_key);
            Ok(result)
        }
    }

    /// two valid cells, move one to another location, and pop that location
    pub(crate) fn replace_cell_within(
        &mut self,
        from_key: TableCellAccess,
        to_key: TableCellAccess,
    ) -> Result<(MultiValue, TableCellAccess), &'static str> {
        // todo cache
        if from_key.column_type() != to_key.column_type() {
            return Err("not on the same column");
        }
        let result = self
            .try_column(from_key.column_type)?
            .remove(to_key.sparse_index)?;
        let access = self
            .try_column(from_key.column_type)?
            .move_value(from_key.sparse_index, to_key.sparse_index)?;
        Ok((result, access))
    }

    /// shallow swap between two valid cells
    pub(crate) fn swap_cell_within(
        &mut self,
        key1: TableCellAccess,
        key2: TableCellAccess,
    ) -> Result<(TableCellAccess, TableCellAccess), &'static str> {
        if key1.column_type() != key2.column_type() {
            return Err("not on the same column");
        }
        // todo cache
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
    pub(crate) fn apply_with_filter(mut vec: AccessColumn, table: &mut MultiTable) -> AccessColumn {
        vec.0.retain(|x| {
            if table
                .get_row(x.sparse_index())
                .unwrap()
                .get_access_from_type(FilterComp::get_id())
                .is_ok()
            {
                return true;
            }
            return false;
        });
        vec
    }
}

pub struct Without<FilterComp: Component>(pub(crate) PhantomData<FilterComp>);
impl<FilterComp: Component> Without<FilterComp> {
    pub(crate) fn apply_without_filter(
        mut vec: AccessColumn,
        table: &mut MultiTable,
    ) -> AccessColumn {
        vec.0.retain(|x| {
            if table
                .get_row(x.sparse_index())
                .unwrap()
                .get_access_from_type(FilterComp::get_id())
                .is_ok()
            {
                return false;
            }
            return true;
        });
        vec
    }
}

pub trait Filter: Sized {
    fn apply_on(vec: AccessColumn, table: &mut MultiTable) -> AccessColumn;
}
impl<FilterComp: Component> Filter for With<FilterComp> {
    fn apply_on(vec: AccessColumn, table: &mut MultiTable) -> AccessColumn {
        With::<FilterComp>::apply_with_filter(vec, table)
    }
}
impl<FilterComp: Component> Filter for Without<FilterComp> {
    fn apply_on(vec: AccessColumn, table: &mut MultiTable) -> AccessColumn {
        Without::<FilterComp>::apply_without_filter(vec, table)
    }
}
impl Filter for () {
    fn apply_on(vec: AccessColumn, table: &mut MultiTable) -> AccessColumn {
        vec
    }
}

pub struct Command<'a> {
    table: &'a mut MultiTable,
}

// TODO: turns the api into wrapper functions of those in impl ComponentTable
impl<'a> Command<'a> {
    pub(crate) fn new(table: &'a mut MultiTable) -> Self {
        Self { table }
    }

    pub fn add_component<C: Component>(&mut self, mut component: C) -> TableCellAccess {
        let comp_type = C::get_id();
        let dst_entity_index = self.table.init_row();
        self.table
            .push_cell(
                dst_entity_index,
                comp_type,
                (&mut component as *mut C).cast::<u8>(),
            )
            .unwrap()
    }

    // key or entity index? usize or generational index?
    pub fn attach_component<C: Component>(
        &mut self,
        key: TableCellAccess,
        mut component: C,
    ) -> Result<TableCellAccess, &'static str> {
        let comp_type = C::get_id();
        // making sure they are different types
        if key.column_type() == comp_type {
            return Err("type not matching");
        }
        let row = self.table.get_row(key.sparse_index())?;
        if row.get_access_from_type(comp_type).is_ok() {
            return Err("type already exists in this row");
        } else {
            let access = self.table.push_cell(
                key.sparse_index(),
                comp_type,
                (&mut component as *mut C).cast::<u8>(),
            )?;
            Ok(access)
        }
    }

    pub fn remove_component<C: Component>(
        &mut self,
        key: TableCellAccess,
    ) -> Result<C, &'static str> {
        if key.column_type() != C::get_id() {
            return Err("type not matching");
        }
        let vec = self.table.pop_cell(key)?;
        // cast this vec into a component
        Ok(unsafe { vec.as_ptr().cast::<C>().as_ref().unwrap().clone() })
    }

    pub fn query<C: Component, F: Filter>(&mut self) -> AccessColumn {
        // let column = self.table.get_column(C::comp_type());
        // match column {
        //     Ok(result) => <F as Filter>::apply_on(result[0], self.table),
        //     // yield empty one
        //     Err(error) => AccessColumn::new_empty(C::comp_type()),
        // }
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

    pub(crate) fn run(&self, table: &mut MultiTable) {
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
    table: MultiTable,
    scheduler: Scheduler,
}

impl ECS {
    pub fn new() -> Self {
        Self {
            table: MultiTable::new(),
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

    fn say_hi(mut command: Command) {
        for player in &mut command.query::<Player, ()>().cast_vec::<Player>() {
            println!("hi, {}", player.0);
        }
    }

    fn remove(mut command: Command) {
        for pl in &mut command.query::<Player, ()>() {
            command.remove_component::<Player>(*pl).unwrap();
        }
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
