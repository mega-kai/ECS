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
use std::alloc::{dealloc, realloc};
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableCellAccess {
    pub(crate) sparse_index: SparseIndex,
    pub(crate) multiptr: MultiPtr,
}
impl TableCellAccess {
    pub(crate) fn new(sparse_index: SparseIndex, multiptr: MultiPtr) -> Self {
        Self {
            sparse_index,
            multiptr,
        }
    }
    pub(crate) fn sparse_index(&self) -> SparseIndex {
        self.sparse_index
    }
    pub(crate) fn column_type(&self) -> MultiCompType {
        self.multiptr.comp_type.clone()
    }
}

// all with the same component type
#[derive(Clone)]
pub struct AccessColumn {
    pub(crate) vec: Vec<TableCellAccess>,
    pub(crate) comp_type: MultiCompType,
}
impl AccessColumn {
    pub(crate) fn new_empty(comp_type: MultiCompType) -> Self {
        Self {
            vec: vec![],
            comp_type,
        }
    }

    pub(crate) fn push(&mut self, access: TableCellAccess) {
        self.vec.push(access)
    }
}
impl<'a> IntoIterator for &'a AccessColumn {
    type Item = &'a TableCellAccess;

    type IntoIter = std::slice::Iter<'a, TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.iter()
    }
}
impl<'a> IntoIterator for &'a mut AccessColumn {
    type Item = &'a mut TableCellAccess;

    type IntoIter = std::slice::IterMut<'a, TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.iter_mut()
    }
}

// all with the same id and and must have diff types
#[derive(Clone)]
pub struct AccessRow {
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

    pub(crate) fn push(&mut self, access: TableCellAccess) -> Result<(), &'static str> {
        if self.get(access.clone().multiptr.comp_type).is_ok() {
            return Err("type duplicated");
        }
        self.access_vec.push(access);
        Ok(())
    }

    pub(crate) fn get(&self, comp_type: MultiCompType) -> Result<TableCellAccess, &'static str> {
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

    pub(crate) fn pop(
        &mut self,
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
            1 => {
                let result = Ok(self.access_vec[final_index].clone());
                self.access_vec.swap_remove(final_index);
                result
            }
            _ => Err("more than one of this type in this row"),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.access_vec.is_empty()
    }

    pub(crate) fn contains_access(&self, key: TableCellAccess) -> bool {
        if key.sparse_index != self.sparse_index {
            return false;
        }
        self.access_vec.contains(&key)
    }

    pub(crate) fn get_current_generation(
        &self,
        comp_type: MultiCompType,
    ) -> Result<Generation, &'static str> {
        Ok(self.get(comp_type)?.multiptr.get_gen())
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

pub(crate) struct Offset(usize);

pub(crate) struct Test {
    comp_type_followed_by_offset_ptr: *mut u8,
    len: usize,
    total_layout: Layout,
}
impl Test {
    unsafe fn extend(&mut self, size: NonZeroUsize) {
        let unit_size = Layout::new::<CompType>().size() + Layout::new::<Offset>().size();
        self.comp_type_followed_by_offset_ptr = realloc(
            self.comp_type_followed_by_offset_ptr,
            self.total_layout,
            self.total_layout.size() + size.get() * unit_size,
        )
    }

    unsafe fn write_comptype(&mut self, index: usize, mut comp_type: CompType) {
        let unit_size = Layout::new::<CompType>().size() + Layout::new::<Offset>().size();
        let comp_size = Layout::new::<CompType>().size();
        let location = self.comp_type_followed_by_offset_ptr.add(index * unit_size);
        let ptr_comp_type = (&mut comp_type as *mut CompType).cast::<u8>();
        copy(ptr_comp_type, location, comp_size);
    }

    unsafe fn write_offset(&mut self, index: usize, mut offset: Offset) {
        let unit_size = Layout::new::<CompType>().size() + Layout::new::<Offset>().size();
        let comp_size = Layout::new::<CompType>().size();
        let offset_size = Layout::new::<Offset>().size();
        let location = self
            .comp_type_followed_by_offset_ptr
            .add(index * unit_size + comp_size);
        let ptr_comp_type = (&mut offset as *mut Offset).cast::<u8>();
        copy(ptr_comp_type, location, offset_size);
    }

    pub(crate) fn new_empty() -> Self {
        let compty_of_compty = CompType::new::<CompType>();
        let compty_of_offset = CompType::new::<Offset>();
        let layout_comptype = Layout::new::<CompType>();
        let layout_offset = Layout::new::<Offset>();
        let (basic_layout, offset_val) = layout_comptype.extend(layout_offset).unwrap();
        let offset_of_offset = Offset(offset_val);
        let ptr = unsafe { std::alloc::alloc(basic_layout) };
        let mut result = Self {
            comp_type_followed_by_offset_ptr: ptr,
            len: 2,
            total_layout: basic_layout,
        };
        unsafe {
            result.write_comptype(0, compty_of_compty);
            result.write_comptype(1, compty_of_offset);
            result.write_offset(0, Offset(0));
            result.write_offset(1, offset_of_offset);
        }
        result
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
        todo!()
    }

    pub(crate) fn new(mut comp_types: Vec<CompType>) -> Result<Self, &'static str> {
        todo!()
    }

    pub(crate) fn get_layout_and_offset(
        &self,
        comp_type: CompType,
    ) -> Result<(Layout, usize), &'static str> {
        todo!()
    }
}
impl Drop for Test {
    fn drop(&mut self) {
        unsafe { dealloc(self.comp_type_followed_by_offset_ptr, self.total_layout) }
    }
}

/// built in sparse index and generation: sparse index offset would always be 0, followed by generation
/// todo refactor this so it can be copy
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
        // self.hashmap_layout_and_offsets.hash(state);
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct MultiPtr {
    ptr: *mut u8,
    comp_type: MultiCompType,
}
impl MultiPtr {
    pub(crate) fn new(ptr: *mut u8, comp_type: MultiCompType) -> Self {
        Self { ptr, comp_type }
    }

    pub(crate) fn cast_multi_value(self) -> MultiValue {
        MultiValue::new(self.ptr, self.comp_type)
    }

    pub(crate) unsafe fn get_sparse_index(&self) -> SparseIndex {
        unsafe { self.ptr.cast::<SparseIndex>().as_ref().unwrap().clone() }
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

    pub(crate) fn to_access(self) -> TableCellAccess {
        TableCellAccess::new(unsafe { self.get_sparse_index() }, self)
    }
}

//-----------------TYPE ERASED VALUES-----------------//
pub(crate) struct Value {
    pub(crate) ptr: *mut u8,
    pub(crate) comp_type: CompType,
}
impl Value {
    pub(crate) fn new(src_ptr: *mut u8, comp_type: CompType) -> Self {
        unsafe {
            let ptr = std::alloc::alloc(comp_type.layout);
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
            std::alloc::dealloc(self.ptr, self.comp_type.layout);
        }
    }
}

pub(crate) struct MultiValue {
    pub(crate) ptr: *mut u8,
    pub(crate) comp_type: MultiCompType,
}
impl MultiValue {
    pub(crate) fn new(src_ptr: *mut u8, comp_type: MultiCompType) -> Self {
        unsafe {
            let ptr = std::alloc::alloc(comp_type.total_layout);
            std::ptr::copy(src_ptr, ptr, comp_type.total_layout.size());
            Self { ptr, comp_type }
        }
    }
}

impl Drop for MultiValue {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.ptr, self.comp_type.total_layout);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
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
        // ptr.cast::<Generation>().unwrap()
        todo!()
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
        MultiPtr::new(ptr, self.ty.clone())
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
        dense_index: DenseIndex,
        ptrs: MultiPtr,
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
            Ok(TableCellAccess::new(sparse_index, raw_dst_ptr))
        }
    }

    pub(crate) fn remove(&mut self, sparse_index: SparseIndex) -> Result<MultiValue, &'static str> {
        let dense_index = self.get_dense_index(sparse_index)?;
        self.sparse[sparse_index] = None;
        let result =
            unsafe { self.replace_multi(dense_index, self.read_multi(DenseIndex(self.len - 1)))? };
        self.len -= 1;
        Ok(result)
    }

    // shallow move
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
            Ok(TableCellAccess::new(to_index, unsafe {
                self.read_multi(dense_index)
            }))
        }
    }

    pub(crate) fn replace(
        &mut self,
        sparse_index: SparseIndex,
        ptrs: MultiPtr,
    ) -> Result<MultiValue, &'static str> {
        unsafe { self.replace_multi(self.get_dense_index(sparse_index)?, ptrs) }
    }

    // shallow swap of two valid cells
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
            TableCellAccess::new(sparse_index1, unsafe { self.read_multi(dense_index2) }),
            TableCellAccess::new(sparse_index2, unsafe { self.read_multi(dense_index1) }),
        ))
    }

    pub(crate) fn get_cell(
        &self,
        sparse_index: SparseIndex,
    ) -> Result<TableCellAccess, &'static str> {
        let ptrs = unsafe { self.read_raw(sparse_index)? };
        Ok(TableCellAccess::new(sparse_index, ptrs))
    }

    pub(crate) fn get_column(&self) -> AccessColumn {
        let mut result = AccessColumn::new_empty(self.ty.clone());
        for index in 0..self.len {
            let dense_index = DenseIndex(index);
            let ptrs = unsafe { self.read_multi(dense_index) };
            result.push(ptrs.to_access());
        }
        result
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
        if self.try_column(comp_type.clone()).is_ok() {
            panic!("type cannot be init twice")
        }
        self.table
            .insert(comp_type.clone(), SparseSet::new(comp_type.clone(), 64));
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
            .entry(comp_type.clone())
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
        let result = self
            .ensure_column(values.comp_type.clone())
            .try_insert(values, sparse_index)?;
        self.row_type_cache
            .get_mut(&sparse_index)
            .ok_or("invalid row")?
            .push(result.clone())?;
        Ok(result)
    }

    pub(crate) fn pop_cell(&mut self, access: TableCellAccess) -> Result<MultiValue, &'static str> {
        self.row_type_cache
            .get_mut(&access.sparse_index)
            .ok_or("invalid row")?
            .pop(access.multiptr.comp_type.clone())?;
        self.try_column(access.multiptr.comp_type.clone())?
            .remove(access.sparse_index)
    }

    pub(crate) fn replace_cell(
        &mut self,
        access: TableCellAccess,
        values: MultiPtr,
    ) -> Result<MultiValue, &'static str> {
        self.row_type_cache
            .get_mut(&access.sparse_index)
            .ok_or("invalid row")?
            .pop(access.multiptr.comp_type.clone())?;
        self.row_type_cache
            .get_mut(&access.sparse_index)
            .ok_or("invalid row")?
            .push(values.clone().to_access())?;
        self.try_column(access.multiptr.comp_type)?
            .replace(access.sparse_index, values)
    }

    //-----------------CELL OPERATION WITHIN TABLE-----------------//

    /// one valid cell move to an empty one, returns the new table cell access
    pub(crate) fn move_cell_within(
        &mut self,
        from_key: TableCellAccess,
        to_index: SparseIndex,
    ) -> Result<TableCellAccess, &'static str> {
        if self
            .try_cell(from_key.multiptr.comp_type.clone(), to_index)
            .is_ok()
        {
            return Err("cell not vacant");
        } else {
            let access = self
                .row_type_cache
                .get_mut(&from_key.sparse_index)
                .ok_or("invalid row")?
                .pop(from_key.multiptr.comp_type.clone())?;
            self.row_type_cache
                .get_mut(&to_index)
                .ok_or("invalid row")?
                .push(access)?;
            let result = self
                .try_column(from_key.multiptr.comp_type.clone())?
                .try_insert(
                    MultiPtr::new(from_key.multiptr.ptr, from_key.multiptr.comp_type.clone()),
                    to_index,
                )?;
            self.pop_cell(from_key)?;
            Ok(result)
        }
    }

    /// two valid cells, move one to another location, and pop that location
    pub(crate) fn replace_cell_within(
        &mut self,
        from_key: TableCellAccess,
        to_key: TableCellAccess,
    ) -> Result<(MultiValue, TableCellAccess), &'static str> {
        if from_key.column_type() != to_key.column_type() {
            return Err("not on the same column");
        }
        let cached_access = self
            .row_type_cache
            .get_mut(&from_key.sparse_index)
            .ok_or("invalid row")?
            .pop(to_key.multiptr.comp_type)?;
        self.row_type_cache
            .get_mut(&to_key.sparse_index)
            .ok_or("invalid row")?
            .push(cached_access)?;

        let result = self
            .try_column(from_key.multiptr.comp_type.clone())?
            .remove(to_key.sparse_index)?;
        let access = self
            .try_column(from_key.multiptr.comp_type)?
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
        let cached_access1 = self
            .row_type_cache
            .get_mut(&key1.sparse_index)
            .ok_or("invalid row")?
            .pop(key1.multiptr.comp_type.clone())?;
        let cached_access2 = self
            .row_type_cache
            .get_mut(&key2.sparse_index)
            .ok_or("invalid row")?
            .pop(key1.multiptr.comp_type.clone())?;
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
    pub(crate) fn apply_with_filter(mut vec: AccessColumn, table: &mut MultiTable) -> AccessColumn {
        todo!()
    }
}

pub struct Without<FilterComp: Component>(pub(crate) PhantomData<FilterComp>);
impl<FilterComp: Component> Without<FilterComp> {
    pub(crate) fn apply_without_filter(
        mut vec: AccessColumn,
        table: &mut MultiTable,
    ) -> AccessColumn {
        todo!()
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
        // let comp_type = C::get_id();
        // let dst_entity_index = self.table.init_row();
        // self.table
        //     .push_cell(
        //         dst_entity_index,
        //         comp_type,
        //         (&mut component as *mut C).cast::<u8>(),
        //     )
        //     .unwrap()
        todo!()
    }

    // key or entity index? usize or generational index?
    pub fn attach_component<C: Component>(
        &mut self,
        key: TableCellAccess,
        mut component: C,
    ) -> Result<TableCellAccess, &'static str> {
        // let comp_type = C::get_id();
        // // making sure they are different types
        // if key.column_type() == comp_type {
        //     return Err("type not matching");
        // }
        // let row = self.table.get_row(key.sparse_index())?;
        // if row.get(comp_type).is_ok() {
        //     return Err("type already exists in this row");
        // } else {
        //     let access = self.table.push_cell(
        //         key.sparse_index(),
        //         comp_type,
        //         (&mut component as *mut C).cast::<u8>(),
        //     )?;
        //     Ok(access)
        // }
        todo!()
    }

    pub fn remove_component<C: Component>(
        &mut self,
        key: TableCellAccess,
    ) -> Result<C, &'static str> {
        // if key.column_type() != C::get_id() {
        //     return Err("type not matching");
        // }
        // let vec = self.table.pop_cell(key)?;
        // // cast this vec into a component
        // Ok(unsafe { vec.as_ptr().cast::<C>().as_ref().unwrap().clone() })
        todo!()
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
