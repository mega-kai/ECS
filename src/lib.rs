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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CompType {
    type_id: TypeId,
    layout: Layout,
}
impl CompType {
    const fn new<C: 'static>() -> Self {
        Self {
            type_id: TypeId::of::<C>(),
            layout: Layout::new::<C>(),
        }
    }
}

#[derive(Clone)]
struct Value {
    ptr: *mut u8,
    comp_type: CompType,
}
impl Value {
    fn new(src_ptr: *mut u8, comp_type: CompType) -> Self {
        unsafe {
            let ptr = alloc(comp_type.layout);
            std::ptr::copy(src_ptr, ptr, comp_type.layout.size());
            Self { ptr, comp_type }
        }
    }

    unsafe fn cast<T: 'static + Clone>(self) -> Result<T, &'static str> {
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

#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
struct SparseIndex(usize);

#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord, Hash)]
struct DenseIndex(usize);

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
struct Generation(usize);
impl Generation {
    fn advance(&mut self) -> Self {
        self.0 += 1;
        *self
    }
}

struct TypeErasedVec {
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

struct DenseVec {
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

    // with tail swap
    fn remove(&mut self, index: DenseIndex) -> (SparseIndex, *mut u8) {
        todo!()
    }
}

struct SparseVec {
    dense_index_vec: TypeErasedVec,
    generation_vec: TypeErasedVec,
}

impl SparseVec {
    fn new() -> Self {
        todo!()
    }

    fn read(&self, sparse_index: SparseIndex) -> Result<DenseIndex, &'static str> {
        todo!()
    }

    // the three functions would panic if sparse index is invalid
    fn toggle_on(&mut self, sparse_index: SparseIndex, dense_index_to_write: DenseIndex) {
        todo!()
    }

    fn toggle_change(&mut self, sparse_index: SparseIndex, dense_index_to_write: DenseIndex) {
        todo!()
    }

    fn toggle_off(&mut self, sparse_index: SparseIndex) {
        todo!()
    }
}

struct Access {
    ptr: *mut u8,
    comp_type: CompType,
    sparse_index: SparseIndex,
    generation: Generation,
}

struct SparseSet {
    comp_type: CompType,
    dense_vec: DenseVec,
    sparse_vec: SparseVec,
}

impl SparseSet {
    fn new(comp_type: CompType, size: usize) -> Self {
        todo!()
    }

    fn is_taken(&self, sparse_index: SparseIndex) -> bool {
        todo!()
    }

    fn read(&self, sparse_index: SparseIndex) -> Result<Access, &'static str> {
        todo!()
    }

    fn write(&mut self, sparse_index: SparseIndex, val: Value) -> Result<Access, &'static str> {
        todo!()
    }

    fn remove(&mut self, sparse_index: SparseIndex) -> Result<Value, &'static str> {
        todo!()
    }

    fn read_all(&self) -> Result<Vec<Access>, &'static str> {
        todo!()
    }
}

struct Table {
    table: HashMap<CompType, SparseSet>,
    bottom_sparse_index: SparseIndex,
}

// TODO: variadic component insertion, probably with tuple
impl Table {
    fn new() -> Self {
        Self {
            table: HashMap::new(),
            bottom_sparse_index: SparseIndex(0),
        }
    }

    //-----------------COLUMN MANIPULATION-----------------//
    fn new_column(&mut self, comp_type: CompType) -> &mut SparseSet {
        if self.try_column(comp_type).is_ok() {
            panic!("type cannot be init twice")
        }
        self.table.insert(comp_type, SparseSet::new(comp_type, 64));
        self.table.get_mut(&comp_type).unwrap()
    }

    fn remove_column(&mut self, comp_type: CompType) -> Option<SparseSet> {
        self.table.remove(&comp_type)
    }

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

    //-----------------ROW MANIPULATION-----------------//
    fn new_row(&mut self) -> SparseIndex {
        let result = self.bottom_sparse_index;
        // todo cache
        self.bottom_sparse_index.0 += 1;
        result
    }

    fn remove_row(&mut self, sparse_index: SparseIndex) -> Result<(), &'static str> {
        todo!()
    }

    //-----------------OPERATIONS-----------------//
    fn write(&mut self, sparse_index: SparseIndex, values: Value) -> Result<Access, &'static str> {
        todo!()
    }

    fn remove(&mut self, access: Access) -> Result<Value, &'static str> {
        todo!()
    }

    fn read(&mut self, sparse_index: SparseIndex) -> Result<Access, &'static str> {
        todo!()
    }

    //-----------------QUERY-----------------//
    fn query() {}
}

struct Command<'a> {
    table: &'a mut Table,
}

// TODO: turns the api into wrapper functions of those in impl ComponentTable
impl<'a> Command<'a> {
    fn new(table: &'a mut Table) -> Self {
        Self { table }
    }

    fn new_entity_with_component(&mut self) {}
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[non_exhaustive]
enum ExecutionFrequency {
    Always,
    Once(bool),
    // Timed(f64, f64),
}
struct System {
    order: usize,
    frequency: ExecutionFrequency,
    func: fn(Command),
}
impl System {
    fn default(func: fn(Command)) -> Self {
        Self {
            order: 0,
            frequency: ExecutionFrequency::Always,
            func,
        }
    }

    fn new(order: usize, frequency: ExecutionFrequency, func: fn(Command)) -> Self {
        Self {
            order,
            frequency,
            func,
        }
    }

    fn run(&self, table: &mut Table) {
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

struct Scheduler {
    new_pool: Vec<System>,
    // waiting: Vec<System>,
    queue: Vec<System>,
}
impl Scheduler {
    fn new() -> Self {
        Self {
            new_pool: vec![],
            // waiting: vec![],
            queue: vec![],
        }
    }

    fn add_system(&mut self, system: System) {
        self.new_pool.push(system);
    }

    fn prepare_queue(&mut self) {
        self.queue.retain(|x| !x.is_once_run());
        if !self.new_pool.is_empty() {
            self.queue.append(&mut self.new_pool);
            self.new_pool.clear();
        }
        self.queue.sort();
    }
}

struct ECS {
    table: Table,
    scheduler: Scheduler,
}

impl ECS {
    fn new() -> Self {
        Self {
            table: Table::new(),
            scheduler: Scheduler::new(),
        }
    }

    fn add_system(&mut self, func: fn(Command), order: usize, once: bool) {
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

    fn tick(&mut self) {
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

    #[derive(Clone, Debug)]
    struct Mana(i32);

    #[derive(Clone, Debug)]
    struct Player(&'static str);
}
