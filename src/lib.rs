#![allow(dead_code, unused_variables, unused_imports, unused_mut)]
#![feature(alloc_layout_extra, map_try_insert)]
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::{Index, IndexMut, Range, RangeBounds};
use std::slice::SliceIndex;
use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    collections::HashMap,
    fmt::Debug,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Location {
    pub(crate) entity_index: usize,
    pub(crate) column_type: CompType,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableCellAccess {
    // pub(crate) assigned_index: usize,
    pub(crate) location: Location,
    pub(crate) access: *mut u8,
    pub(crate) generation: usize,
}
impl TableCellAccess {
    pub(crate) fn new(
        entity_index: usize,
        column_type: CompType,
        access: *mut u8,
        generation: usize,
    ) -> Self {
        Self {
            location: Location {
                entity_index,
                column_type,
            },
            access,
            generation,
        }
    }
    pub(crate) fn entity_index(&self) -> usize {
        self.location.entity_index
    }
    pub(crate) fn column_type(&self) -> CompType {
        self.location.column_type
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
            .map(|x| unsafe { x.access.cast::<C>().as_mut().unwrap() })
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
    pub(crate) entity_index: usize,
}
impl AccessRow {
    pub(crate) fn new(access_vec: Vec<TableCellAccess>, entity_index: usize) -> Self {
        Self {
            access_vec,
            entity_index,
        }
    }

    pub(crate) fn new_empty(entity_index: usize) -> Self {
        Self {
            access_vec: vec![],
            entity_index,
        }
    }

    pub(crate) fn get_access_from_type(
        &self,
        comp_type: CompType,
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
        comp_type: CompType,
    ) -> Result<usize, &'static str> {
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

pub(crate) struct SparseIndex(Option<usize>);
pub(crate) struct Generation(usize);

/// built in sparse index and generation: sparse index offset would always be 0
pub(crate) struct MultiCompType {
    hashmap_layout_and_offsets: HashMap<CompType, (Layout, usize)>,
    total_layout: Layout,
    generation_offset: usize,
}
impl MultiCompType {
    pub(crate) fn new(mut comp_types: Vec<CompType>) -> Self {
        if comp_types.is_empty() {
            panic!("empty types")
        }
        for ty in comp_types.iter() {
            if ty.layout.size() == 0 {
                panic!("ZST included in this vec")
            }
        }
        let mut layout_and_offsets = HashMap::new();
        // first
        let mut total_layout = Layout::new::<SparseIndex>();
        let (with_generation_layout, generation_offset) =
            total_layout.extend(Layout::new::<Generation>()).unwrap();
        for ty in comp_types {
            let (layout, offset) = total_layout.extend(ty.layout).unwrap();
            layout_and_offsets.insert(ty, (layout, offset));
        }

        Self {
            hashmap_layout_and_offsets: layout_and_offsets,
            total_layout,
            generation_offset,
        }
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

//-----------------SINGLE TYPE ERASED POINTERS-----------------//
pub(crate) struct OwningPtr {
    pub(crate) ptr: *mut u8,
    pub(crate) comp_type: CompType,
}
pub(crate) struct SharedPtr {
    pub(crate) ptr: *const u8,
    pub(crate) comp_type: CompType,
}

pub(crate) struct MutPtr {
    pub(crate) ptr: *mut u8,
    pub(crate) comp_type: CompType,
}

//-----------------MULTI TYPE ERASED POINTERS-----------------//
pub(crate) struct MultiOwningPtr(Vec<OwningPtr>);
pub(crate) struct MultiSharedPtr(Vec<SharedPtr>);
pub(crate) struct MultiMutPtr(Vec<MutPtr>);

//-----------------TYPE ERASED VALUES-----------------//
pub(crate) struct SingleValue {
    pub(crate) val: Vec<u8>,
    pub(crate) comp_type: CompType,
}

pub(crate) struct MultiValue(Vec<SingleValue>);

pub(crate) struct SparseSet {
    ty: MultiCompType,
    data_heap_ptr: *mut u8,
    pub(crate) capacity: usize,
    pub(crate) len: usize,
    // usize == dense_index
    pub(crate) sparse: Vec<Option<usize>>,
}

// todo: make operations unsafe as they should be
impl SparseSet {
    pub(crate) fn new(comp_types: Vec<CompType>, size: usize) -> Self {
        let result_comp_types = MultiCompType::new(comp_types);

        let data_heap_ptr =
            unsafe { std::alloc::alloc(result_comp_types.total_layout().repeat(size).unwrap().0) };
        Self {
            ty: result_comp_types,
            data_heap_ptr,
            capacity: size,
            len: 0,
            sparse: vec![None; size],
        }
    }

    //-----------------HELPERS-----------------//
    /// must ensure dense_index is valid first
    fn get_dense_ptr(
        &self,
        dense_index: usize,
        comp_type: CompType,
    ) -> Result<*mut u8, &'static str> {
        let (_, offset) = self.ty.get_layout_and_offset(comp_type)?;
        unsafe {
            Ok(self
                .data_heap_ptr
                .add(self.ty.total_layout().size() * dense_index + offset))
        }
    }

    fn get_dense_index(&self, sparse_index: usize) -> Result<usize, &'static str> {
        if sparse_index >= self.sparse.len() {
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

    /// get_dense_index + get_dense_ptr
    pub(crate) fn get_single(
        &self,
        entity_id: usize,
        comp_type: CompType,
    ) -> Result<*mut u8, &'static str> {
        let dense_index = self.get_dense_index(entity_id)?;
        Ok(self.get_dense_ptr(dense_index, comp_type)?)
    }

    pub(crate) fn get_all(&self, comp_type: CompType) -> Result<Vec<*mut u8>, &'static str> {
        let mut result_vec: Vec<*mut u8> = vec![];
        for index in 0..self.len {
            let ptr = self.get_dense_ptr(index, comp_type)?;
            result_vec.push(ptr);
        }
        Ok(result_vec)
    }

    //-----------------OPERATIONS-----------------//

    unsafe fn write_single(&self, dense_index: usize, owning_ptr: OwningPtr) {
        let dst_ptr = self.data_heap_ptr.add(
            dense_index * self.ty.total_layout().size()
                + self
                    .ty
                    .get_layout_and_offset(owning_ptr.comp_type)
                    .unwrap()
                    .1,
        );
        std::ptr::copy(owning_ptr.ptr, dst_ptr, owning_ptr.comp_type.layout.size());
    }

    /// on caller to ensure that all types match with comp types with the same order
    /// todo handle generation since in this way the generation is passed from the caller
    pub(crate) fn add(
        &mut self,
        ptrs: Vec<*mut u8>,
        entity_id: usize,
    ) -> Result<*mut u8, &'static str> {
        if self.get_dense_index(entity_id).is_ok() {
            return Err("cell taken");
        }

        if ptrs.len() != self.ty.len() {
            return Err("wrong size of ptrs");
        }

        if entity_id >= self.sparse.len() {
            self.sparse.resize(entity_id + 1, None);
        }

        if self.len >= self.capacity {
            self.double_dense_cap();
        }

        self.sparse[entity_id] = Some(self.len);
        let raw_dst_ptr = self.get_dense_ptr(self.len, CompType::new::<SparseIndex>())?;

        for (index, ptr) in ptrs.into_iter().enumerate() {
            // unsafe {
            //     self.write_single(self.len, ptr, self.comp_types.types[index]);
            // }
            todo!()
        }
        self.len += 1;
        Ok(raw_dst_ptr)
    }

    pub(crate) unsafe fn replace_single(
        &self,
        src_ptr: *mut u8,
        dense_index: usize,
        comp_type: CompType,
    ) -> Vec<u8> {
        let mut vec = vec![0u8; comp_type.layout.size()];
        let vec_ptr = vec.as_mut_ptr();
        let ptr_in_dense = self.data_heap_ptr.add(
            dense_index * self.ty.total_layout().size()
                + self.ty.get_layout_and_offset(comp_type).unwrap().1,
        );
        std::ptr::copy(ptr_in_dense, vec_ptr, comp_type.layout.size());
        std::ptr::copy(src_ptr, ptr_in_dense, comp_type.layout.size());
        vec
    }

    pub(crate) fn replace(
        &self,
        ptrs: Vec<*mut u8>,
        dst_entity_index: usize,
    ) -> Result<Vec<Vec<u8>>, &'static str> {
        let dense_index = self.get_dense_index(dst_entity_index)?;
        let mut vec: Vec<Vec<u8>> = vec![];
        for (index, ptr) in ptrs.into_iter().enumerate() {
            // vec.push(unsafe {
            //     self.replace_single(ptr, dense_index, self.comp_types.types[index])
            // });
            todo!()
        }
        Ok(vec)
    }

    unsafe fn remove_single(&self, dense_index: usize, comp_type: CompType) -> Vec<u8> {
        let mut vec = vec![0u8; comp_type.layout.size()];
        let vec_ptr = vec.as_mut_ptr();
        let src_ptr = self.data_heap_ptr.add(
            dense_index * self.ty.total_layout().size()
                + self.ty.get_layout_and_offset(comp_type).unwrap().1,
        );
        std::ptr::copy(src_ptr, vec_ptr, comp_type.layout.size());
        vec
    }

    pub(crate) fn remove(&mut self, entity_id: usize) -> Result<Vec<Vec<u8>>, &'static str> {
        let dense_index = self.get_dense_index(entity_id)?;
        self.sparse[entity_id] = None;
        let mut vec: Vec<Vec<u8>> = vec![];
        for index in 0..self.ty.len() {
            // vec.push(unsafe { self.remove_single(dense_index, self.comp_types.types[index]) });
            todo!()
        }
        Ok(vec)
    }

    // "shallow swap"
    pub(crate) fn swap_within(&mut self, index1: usize, index2: usize) -> Result<(), &'static str> {
        let dense_index1 = self.get_dense_index(index1)?;
        let dense_index2 = self.get_dense_index(index2)?;
        self.sparse[index1] = Some(dense_index2);
        self.sparse[index2] = Some(dense_index1);
        Ok(())
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
    // with this column containing both component and generational index
    table: HashMap<CompType, SparseSet>,
    // all valid cells in this row; indexed by entity index
    row_type_cache: HashMap<usize, AccessRow>,
    bottom_row_id: usize,
}

// TODO: generational indices
// TODO: make this table multiple type compactable
// TODO: cache all the comp types of all the rows, update the cache upon add/attach/remove/swap
// TODO: refactoring api to location/pointer/generation seperate
// TODO: incorporate all the query filter methods within the table api, making it a more proper table data structure
// TODO: variadic component insertion, probably with tuple
// TODO: concept of type erased ptr(shared, mut, owning) that includes comp type, and also a wrapper type; memory safety of TableCellAccess
impl MultiTable {
    pub(crate) fn new() -> Self {
        Self {
            table: HashMap::new(),
            row_type_cache: HashMap::new(),
            bottom_row_id: 0,
        }
    }

    //-----------------COLUMN MANIPULATION-----------------//
    pub(crate) fn init_column(&mut self, comp_type: CompType) -> &mut SparseSet {
        if self.try_column(comp_type).is_ok() {
            panic!("type cannot be init twice")
        }
        self.table
            .insert(comp_type, SparseSet::new(vec![comp_type], 64));
        self.table.get_mut(&comp_type).unwrap()
    }

    pub(crate) fn get_column(&mut self, comp_type: CompType) -> Result<Vec<*mut u8>, &'static str> {
        Ok(self.try_column(comp_type)?.get_all(comp_type)?)
    }

    pub(crate) fn pop_column(&mut self, comp_type: CompType) -> Option<SparseSet> {
        self.table.remove(&comp_type)
    }

    //-----------------ROW MANIPULATION-----------------//
    pub(crate) fn init_row(&mut self) -> usize {
        self.bottom_row_id += 1;
        self.row_type_cache.insert(
            self.bottom_row_id - 1,
            AccessRow::new_empty(self.bottom_row_id - 1),
        );
        self.bottom_row_id - 1
    }

    pub(crate) fn get_row(&mut self, entity_index: usize) -> Result<AccessRow, &'static str> {
        // since init row ensures the existence of all row cache
        let cache = self
            .row_type_cache
            .get(&entity_index)
            .ok_or("index overflow")?;
        Ok(cache.clone())
    }

    //-----------------CELL MANIPULATION HELPERS-----------------//
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
            .or_insert(SparseSet::new(vec![comp_type], 64))
    }

    /// bypassing generational index; if empty, returns err, else returns raw ptr
    pub(crate) fn is_valid_raw(
        &mut self,
        entity_index: usize,
        comp_type: CompType,
    ) -> Result<*mut u8, &'static str> {
        Ok(self
            .try_column(comp_type)?
            .get_single(entity_index, comp_type)?)
    }

    //-----------------CELL IO OPERATION-----------------//

    // if not column init, init it automatically
    pub(crate) fn push_cell(
        &mut self,
        dst_entity_index: usize,
        comp_type: CompType,
        ptr: *mut u8,
    ) -> Result<TableCellAccess, &'static str> {
        let ptr = self
            .ensure_column(comp_type)
            .add(vec![ptr], dst_entity_index)?;
        // todo cache
        Ok(TableCellAccess::new(dst_entity_index, comp_type, ptr, 0))
    }

    pub(crate) fn pop_cell(&mut self, key: TableCellAccess) -> Result<Vec<u8>, &'static str> {
        // todo cache
        // todo fix this indexing jankiness
        let thing = self
            .try_column(key.column_type())?
            .remove(key.entity_index())?[1]
            .clone();
        Ok(thing)
    }

    /// write and return the old one in a series of bytes in a vector
    /// it is on the caller to ensure they are the same type, else it's UB
    pub(crate) fn replace_cell(
        &mut self,
        key: TableCellAccess,
        ptr: *mut u8,
    ) -> Result<Vec<u8>, &'static str> {
        // todo cache
        let column = self.try_column(key.column_type())?;
        Ok(column.replace(vec![ptr], key.entity_index())?[1].clone())
    }

    //-----------------CELL OPERATION WITHIN TABLE-----------------//

    /// one valid cell move to an empty one, returns the new table cell access
    pub(crate) fn move_cell_within(
        &mut self,
        from_key: TableCellAccess,
        to_index: usize,
    ) -> Result<TableCellAccess, &'static str> {
        if self.is_valid_raw(to_index, from_key.column_type()).is_err() {
            let ptr = self
                .try_column(from_key.column_type())?
                .add(vec![from_key.access], to_index)?;
            // todo cache
            let new_access = TableCellAccess::new(to_index, from_key.column_type(), ptr, 0);
            Ok(new_access)
        } else {
            Err("dst isn't empty")
        }
    }

    /// two valid cells, move one to another location, and pop that location
    pub(crate) fn replace_cell_within(
        &mut self,
        from_key: TableCellAccess,
        to_key: TableCellAccess,
    ) -> Result<Vec<u8>, &'static str> {
        // todo cache
        if from_key.column_type() != to_key.column_type() {
            return Err("not on the same column");
        }
        let column = self.try_column(from_key.column_type())?;
        let ptr = column.get_single(from_key.entity_index(), from_key.column_type())?;
        let vec = column.replace(vec![ptr], to_key.entity_index())?[1].clone();
        Ok(vec)
    }

    /// shallow swap between two valid cells
    pub(crate) fn swap_cell_within(
        &mut self,
        key1: TableCellAccess,
        key2: TableCellAccess,
    ) -> Result<(), &'static str> {
        if key1.column_type() != key2.column_type() {
            return Err("not on the same column");
        }
        // todo cache
        self.try_column(key1.column_type())?
            .swap_within(key1.entity_index(), key2.entity_index())
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
                .get_row(x.entity_index())
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
                .get_row(x.entity_index())
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
        let row = self.table.get_row(key.entity_index())?;
        if row.get_access_from_type(comp_type).is_ok() {
            return Err("type already exists in this row");
        } else {
            let access = self.table.push_cell(
                key.entity_index(),
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
