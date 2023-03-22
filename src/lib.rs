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
pub struct Ptr {
    ptr: *mut u8,
    comp_type: CompType,
    sparse_index: SparseIndex,
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
pub(crate) struct PtrColumn {
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

//-----------------GENERATION-----------------//
#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub(crate) struct Generation(usize);
impl Generation {
    pub(crate) fn advance(&mut self) -> Self {
        self.0 += 1;
        *self
    }
}

//-----------------SPARSE/DENSE INDEX-----------------//
#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
pub(crate) struct SparseIndex(usize);

#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
pub(crate) struct DenseIndex(usize);

//----------------BASIC DATA STRUCTURES------------------//
pub(crate) struct TypeErasedVec {
    ptr: *mut u8,
    len: usize,
    cap: usize,
    comp_type: CompType,
}

impl TypeErasedVec {
    fn new(comp_type: CompType, size: usize) -> Self {
        let ptr = unsafe { alloc(comp_type.layout.repeat(size).unwrap().0) };
        Self {
            ptr,
            len: 0,
            cap: size,
            comp_type,
        }
    }

    fn read(&mut self, index: usize) -> *mut u8 {
        todo!()
    }

    fn write(&mut self, index: usize, ptrs: *mut u8) {
        // would automatically adjust length by recursively doubling its own length
        todo!()
    }

    fn push(&mut self, ptrs: *mut u8) {
        todo!()
    }

    fn double_len(&mut self) {
        todo!()
    }
}

impl Drop for TypeErasedVec {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr, self.comp_type.layout.repeat(self.cap).unwrap().0);
        }
    }
}

pub(crate) struct DenseVec {
    sparse_index_vec: TypeErasedVec,
    comp_vec: TypeErasedVec,
}

impl DenseVec {
    fn new(comp_type: CompType, size: usize) -> Self {
        let result = Self {
            sparse_index_vec: TypeErasedVec::new(CompType::new::<SparseIndex>(), size),
            comp_vec: TypeErasedVec::new(comp_type, size),
        };
        todo!()
    }

    fn read(&mut self, index: DenseIndex) -> (SparseIndex, *mut u8) {
        todo!()
    }

    fn write(&mut self, index: DenseIndex, content: (SparseIndex, *mut u8)) {
        todo!()
    }

    fn push(&mut self, content: (SparseIndex, *mut u8)) {
        todo!()
    }

    // with tail swap
    fn remove(&mut self, index: DenseIndex) -> (SparseIndex, *mut u8) {
        todo!()
    }
}

pub(crate) struct SparseVec {
    dense_index_vec: TypeErasedVec,
    generation_vec: TypeErasedVec,
}

impl SparseVec {
    fn new() -> Self {
        todo!()
    }

    // the three functions would panic if sparse index is invalid
    fn toggle_on(&mut self, sparse_index: SparseIndex, dense_index_to_write: DenseIndex) {
        todo!()
    }

    fn toggle_off(&mut self, sparse_index: SparseIndex) {
        todo!()
    }

    fn toggle_change(&mut self, sparse_index: SparseIndex, dense_index_to_write: DenseIndex) {
        todo!()
    }
}

//----------------SPARSE SET------------------//
pub(crate) struct SparseSet {
    comp_type: CompType,

    dense_vec: DenseVec,
    sparse_vec: SparseVec,
}

impl SparseSet {
    pub(crate) fn new(comp_type: CompType, size: usize) -> Self {
        todo!()
    }

    //-----------------MAIN API-----------------//
    pub(crate) fn is_taken(&self, sparse_index: SparseIndex) -> bool {
        todo!()
    }

    pub(crate) fn insert(
        &mut self,
        sparse_index: SparseIndex,
        ptrs: Ptr,
    ) -> Result<Ptr, &'static str> {
        todo!()
    }

    pub(crate) fn remove(&mut self, sparse_index: SparseIndex) -> Result<Value, &'static str> {
        todo!()
    }

    // shallow move
    pub(crate) fn move_value(
        &mut self,
        from_index: SparseIndex,
        to_index: SparseIndex,
    ) -> Result<Ptr, &'static str> {
        todo!()
    }

    pub(crate) fn replace(
        &mut self,
        sparse_index: SparseIndex,
        ptrs: Ptr,
    ) -> Result<Value, &'static str> {
        todo!()
    }

    // shallow swap of two valid cells
    pub(crate) fn swap_within(
        &mut self,
        sparse_index1: SparseIndex,
        sparse_index2: SparseIndex,
    ) -> Result<(Ptr, Ptr), &'static str> {
        todo!()
    }

    pub(crate) fn get_cell(&self, sparse_index: SparseIndex) -> Result<Ptr, &'static str> {
        todo!()
    }

    pub(crate) fn get_column(&self) -> PtrColumn {
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
    pub(crate) fn new_column(&mut self, comp_type: CompType) -> &mut SparseSet {
        if self.try_column(comp_type).is_ok() {
            panic!("type cannot be init twice")
        }
        self.table.insert(comp_type, SparseSet::new(comp_type, 64));
        self.table.get_mut(&comp_type).unwrap()
    }

    pub(crate) fn get_column(&mut self, comp_type: CompType) -> Result<PtrColumn, &'static str> {
        Ok(self.try_column(comp_type)?.get_column())
    }

    pub(crate) fn remove_column(&mut self, comp_type: CompType) -> Option<SparseSet> {
        self.table.remove(&comp_type)
    }

    //-----------------ROW MANIPULATION-----------------//
    pub(crate) fn new_row(&mut self) -> SparseIndex {
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

    //-----------------CELL IO OPERATION-----------------//

    // if not column init, init it automatically
    pub(crate) fn push_cell(
        &mut self,
        sparse_index: SparseIndex,
        // todo: maybe Value??
        values: Ptr,
    ) -> Result<Ptr, &'static str> {
        todo!()
    }

    pub(crate) fn pop_cell(&mut self, access: Ptr) -> Result<Value, &'static str> {
        todo!()
    }

    pub(crate) fn replace_cell(
        &mut self,
        access: Ptr,
        // todo maybe value?
        values: Ptr,
    ) -> Result<Value, &'static str> {
        todo!()
    }

    //-----------------CELL OPERATION WITHIN TABLE-----------------//

    /// one valid cell move to an empty one, returns the new table cell access
    pub(crate) fn move_cell_within(
        &mut self,
        from_key: Ptr,
        to_index: SparseIndex,
    ) -> Result<Ptr, &'static str> {
        todo!()
    }

    /// two valid cells, move one to another location, and pop that location
    pub(crate) fn replace_cell_within(
        &mut self,
        from_key: Ptr,
        to_key: Ptr,
    ) -> Result<(Value, Ptr), &'static str> {
        todo!()
    }

    /// shallow swap between two valid cells
    pub(crate) fn swap_cell_within(
        &mut self,
        key1: Ptr,
        key2: Ptr,
    ) -> Result<(Ptr, Ptr), &'static str> {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ExecutionFrequency {
    Always,
    Once(bool),
    // Timed(f64, f64),
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

    // pub fn query<C: Component, F: Filter>(&mut self) -> PtrColumn {
    //     todo!()
    // }
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
