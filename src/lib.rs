#![allow(dead_code, unused_variables, unreachable_code, unused_mut)]
#![feature(
    alloc_layout_extra,
    map_try_insert,
    core_intrinsics,
    const_trait_impl,
    const_type_id,
    const_mut_refs,
    const_type_name,
    portable_simd,
    array_zip,
    slice_as_chunks
)]

use hashbrown::HashMap;
use std::alloc::{alloc, dealloc, realloc};
use std::hash::Hash;
use std::ops::{BitAnd, BitOr, BitXor, Not};
use std::simd::Simd;
use std::slice::from_raw_parts_mut;
use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    fmt::Debug,
};

//-----------------STORAGE-----------------//
type SparseIndex = usize;
type DenseIndex = usize;
const NULL: Filter = Filter::from::<()>();

#[cfg(target_pointer_width = "64")]
const MASK: Simd<usize, 64> = Simd::from_array([9223372036854775808; 64]);
#[cfg(target_pointer_width = "32")]
const MASK: Simd<usize, 64> = Simd::from_array([2147483648; 64]);

trait Tuple: Sized {
    fn insert(
        self,
        table: &mut Table,
        sparse_index: Option<SparseIndex>,
    ) -> Result<(), &'static str>;
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str>;
}

impl<C0> Tuple for (C0,)
where
    C0: 'static + Clone + Sized,
{
    fn insert(
        self,
        table: &mut Table,
        sparse_index: Option<SparseIndex>,
    ) -> Result<(), &'static str> {
        table.insert(sparse_index, self.0)?;
        Ok(())
    }
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str> {
        Ok((table.remove::<C0>(sparse_index)?,))
    }
}
impl<C0, C1> Tuple for (C0, C1)
where
    C0: 'static + Clone + Sized,
    C1: 'static + Clone + Sized,
{
    fn insert(
        self,
        table: &mut Table,
        sparse_index: Option<SparseIndex>,
    ) -> Result<(), &'static str> {
        table.insert(sparse_index, self.0)?;
        table.insert(sparse_index, self.1)?;
        Ok(())
    }
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str> {
        Ok((
            table.remove::<C0>(sparse_index)?,
            table.remove::<C1>(sparse_index)?,
        ))
    }
}
impl<C0, C1, C2> Tuple for (C0, C1, C2)
where
    C0: 'static + Clone + Sized,
    C1: 'static + Clone + Sized,
    C2: 'static + Clone + Sized,
{
    fn insert(
        self,
        table: &mut Table,
        sparse_index: Option<SparseIndex>,
    ) -> Result<(), &'static str> {
        table.insert(sparse_index, self.0)?;
        table.insert(sparse_index, self.1)?;
        table.insert(sparse_index, self.2)?;
        Ok(())
    }
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str> {
        Ok((
            table.remove::<C0>(sparse_index)?,
            table.remove::<C1>(sparse_index)?,
            table.remove::<C2>(sparse_index)?,
        ))
    }
}
impl<C0, C1, C2, C3> Tuple for (C0, C1, C2, C3)
where
    C0: 'static + Clone + Sized,
    C1: 'static + Clone + Sized,
    C2: 'static + Clone + Sized,
    C3: 'static + Clone + Sized,
{
    fn insert(
        self,
        table: &mut Table,
        sparse_index: Option<SparseIndex>,
    ) -> Result<(), &'static str> {
        table.insert(sparse_index, self.0)?;
        table.insert(sparse_index, self.1)?;
        table.insert(sparse_index, self.2)?;
        table.insert(sparse_index, self.3)?;
        Ok(())
    }
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str> {
        Ok((
            table.remove::<C0>(sparse_index)?,
            table.remove::<C1>(sparse_index)?,
            table.remove::<C2>(sparse_index)?,
            table.remove::<C3>(sparse_index)?,
        ))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CompType {
    type_id: TypeId,
    layout: Layout,
}
impl CompType {
    const fn new<C: 'static + Clone + Sized>() -> Self {
        Self {
            type_id: TypeId::of::<C>(),
            layout: Layout::new::<C>(),
        }
    }
}
impl Debug for CompType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let trimmed = type_name::<Self>().split("::");
        write!(f, "{}", trimmed.last().unwrap())
    }
}

struct TypeErasedVec {
    ptr: *mut u8,
    len: usize,
    cap: usize,
    comp_type: CompType,
}
impl TypeErasedVec {
    fn new_empty<C: 'static + Clone + Sized>(cap: usize) -> Self {
        assert!(cap != 0, "zero capacity is not allowed");
        let comp_type = CompType::new::<C>();
        assert!(comp_type.layout.size() != 0, "zst");
        let ptr = unsafe { alloc(comp_type.layout.repeat(cap).unwrap().0) };
        Self {
            ptr,
            len: 0,
            cap,
            comp_type,
        }
    }

    fn push<C: 'static + Clone + Sized>(&mut self, grow_len: usize, value: C) {
        self.ensure_cap(grow_len + self.len);
        let old_len = self.len;
        self.len += grow_len;
        self.as_slice::<C>()[old_len..self.len - 1].fill(value);
    }

    fn ensure_cap(&mut self, cap: usize) {
        if cap > self.cap {
            self.ptr = unsafe {
                realloc(
                    self.ptr,
                    self.comp_type.layout.repeat(self.len).unwrap().0,
                    self.cap * 2 * self.comp_type.layout.size(),
                )
            };
            self.cap *= 2;
            self.ensure_cap(cap);
        }
    }

    fn as_slice<C>(&self) -> &mut [C] {
        unsafe { from_raw_parts_mut(self.ptr as *mut C, self.len) }
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
    fn new_empty<C: 'static + Clone + Sized>() -> Self {
        let result = Self {
            sparse_index_vec: TypeErasedVec::new_empty::<SparseIndex>(64),
            comp_vec: TypeErasedVec::new_empty::<C>(64),
        };
        result
    }

    fn push<C: 'static + Clone + Sized>(&mut self, content: (SparseIndex, C)) -> DenseIndex {
        self.comp_vec.push(1, content.1);
        self.sparse_index_vec.push(1, content.0);
        self.comp_vec.len - 1
    }

    // with tail swap
    fn remove<C: 'static + Clone + Sized>(
        &mut self,
        dense_index: DenseIndex,
    ) -> Result<(SparseIndex, C), &'static str> {
        let value = (
            self.as_slice()[dense_index],
            self.comp_vec.as_slice::<C>()[dense_index].clone(),
        );

        self.as_slice().copy_within(
            self.sparse_index_vec.len - 1..self.sparse_index_vec.len - 1,
            dense_index,
        );
        self.comp_vec.as_slice::<C>()[dense_index] =
            self.comp_vec.as_slice::<C>().last().unwrap().clone();

        self.comp_vec.len -= 1;
        self.sparse_index_vec.len -= 1;
        Ok(value)
    }

    fn as_slice(&self) -> &mut [SparseIndex] {
        self.sparse_index_vec.as_slice()
    }
}

struct SparseVec {
    dense_index_vec: TypeErasedVec,
}

impl SparseVec {
    fn new() -> Self {
        let mut thing = Self {
            dense_index_vec: TypeErasedVec::new_empty::<DenseIndex>(64),
        };
        thing.dense_index_vec.push::<DenseIndex>(64, 0);
        thing
    }

    // note that if it's not empty it will have an on bit at the start
    fn as_slice(&self) -> &mut [DenseIndex] {
        self.dense_index_vec.as_slice()
    }

    fn as_simd(&self) -> &mut [Simd<DenseIndex, 64>] {
        unsafe {
            from_raw_parts_mut(
                self.dense_index_vec.ptr as *mut Simd<DenseIndex, 64>,
                self.dense_index_vec.len / 64,
            )
        }
    }
}

struct SparseSet {
    dense_vec: DenseVec,
    sparse_vec: SparseVec,
}

impl SparseSet {
    fn new<C: 'static + Clone + Sized>() -> Self {
        Self {
            dense_vec: DenseVec::new_empty::<C>(),
            sparse_vec: SparseVec::new(),
        }
    }

    fn insert<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: SparseIndex,
        value: C,
    ) -> Result<(), &'static str> {
        if self.sparse_vec.as_slice()[sparse_index] == 0 {
            let dense = self.dense_vec.push((sparse_index, value));
            self.sparse_vec.as_slice()[sparse_index] = dense;
            Ok(())
        } else {
            Err("sparse taken")
        }
    }

    // todo handle tail swap
    fn remove<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: SparseIndex,
    ) -> Result<C, &'static str> {
        let dense_index = self.sparse_vec.as_slice()[sparse_index];
        if dense_index != 0 {
            self.sparse_vec.as_slice()[sparse_index] = 0;
            let val = self.dense_vec.remove::<C>(dense_index)?.1;
            self.sparse_vec.as_slice()[self.dense_vec.as_slice()[dense_index]] = dense_index;
            Ok(val)
        } else {
            Err("sparse empty")
        }
    }
}

// each table slice correspond to one filter; plus there's no such thing is aggressive caching, since ^ and | exists within the filter
struct SubTable {
    // local index -> ptr to actual data
    content: HashMap<CompType, TypeErasedVec>,
}
impl SubTable {
    fn new() -> Self {
        Self {
            content: HashMap::new(),
        };
        let vec = TypeErasedVec::new_empty::<Option<*mut u8>>(64);
        // populate the slice
        todo!()
    }
}

struct Table {
    global_table: HashMap<CompType, SparseSet>,
    bottom_sparse_index: SparseIndex,
    sub_tables: HashMap<Filter, SubTable>,
}

impl Table {
    fn new() -> Self {
        todo!()
    }

    fn insert<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: Option<SparseIndex>,
        value: C,
    ) -> Result<(), &'static str> {
        let sparse = if let Some(val) = sparse_index {
            val
        } else {
            self.bottom_sparse_index += 1;
            self.bottom_sparse_index - 1
        };
        self.global_table
            .entry(CompType::new::<C>())
            .or_insert(SparseSet::new::<C>())
            .insert(sparse, value)
        // todo, update cache
    }

    fn remove<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: SparseIndex,
    ) -> Result<C, &'static str> {
        self.global_table
            .get_mut(&CompType::new::<C>())
            .ok_or("comp type not in table")?
            .remove(sparse_index)
        // todo, pudate cache
    }

    fn raw_query(&self, filter: &Filter) -> SubTable {
        todo!()
    }

    fn query<T: Tuple>(&mut self, filter: &Filter) -> IterMut<T> {
        if filter.is_null() {
            // get all the components of this type
            todo!()
        }
        let access = if let Some(val) = self.sub_tables.get(filter) {
            val
        } else {
            self.sub_tables
                .insert(filter.clone(), self.raw_query(filter));
            self.sub_tables.get(filter).unwrap()
        };
        IterMut::new(access)
    }
}

struct IterMut<'a, T: Tuple>(&'a mut T);
impl<'a, T: Tuple> IterMut<'a, T> {
    fn new_empty() -> Self {
        todo!()
    }

    fn new(sub_table: &SubTable) -> Self {
        todo!()
    }
}
impl<'a, T: Tuple> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

//-----------------FILTER-----------------//
// TODO: figure out the relationship between filters and its "sub/super filters"
#[derive(Eq, Clone)]
struct Filter {
    // [n][0..2] = negation flag of all the previous evaluation; [n][2..4] = operation other than Not; [n][4..6] = negation flag for self(always single)
    ops_flags: [u8; 9],
    ids: [u64; 9],
    #[cfg(debug_assertions)]
    names: [&'static str; 8],
}
impl Hash for Filter {
    fn hash<H: ~const std::hash::Hasher>(&self, state: &mut H) {
        // first element in u64 array is the operation result of all the type id;
        // first element in u8 array is the number of elements in this whole structure; if it's singleton then the u64 would be just the type id
        self.ops_flags[0].hash(state);
        self.ids[0].hash(state);
    }
}
impl PartialEq for Filter {
    fn eq(&self, other: &Self) -> bool {
        self.ops_flags[0] == other.ops_flags[0] && self.ids[0] == other.ids[0]
    }
}
impl Filter {
    const fn from<C: Clone + 'static>() -> Self {
        Self {
            ops_flags: [0; 9],
            ids: [0; 9],
            names: [&""; 8],
        }
    }

    fn get_num(&self) -> usize {
        todo!()
    }

    fn is_null(&self) -> bool {
        todo!()
    }
}
impl BitAnd for Filter {
    type Output = Filter;
    fn bitand(self, rhs: Self) -> Self::Output {
        // determine whether or not it's a singleton
        todo!()
    }
}
impl BitOr for Filter {
    type Output = Filter;
    fn bitor(self, rhs: Self) -> Self::Output {
        todo!()
    }
}
impl BitXor for Filter {
    type Output = Filter;
    fn bitxor(self, rhs: Self) -> Self::Output {
        todo!()
    }
}
impl Not for Filter {
    type Output = Filter;
    fn not(self) -> Self::Output {
        todo!()
    }
}

// a filter can be an archetype or a subset of an archetype

//-----------------COMMAND-----------------//
struct Command<'a> {
    table: &'a mut Table,
    system: &'a mut System,
}
impl<'a> Command<'a> {
    fn new(table: &'a mut Table, system: &'a mut System) -> Self {
        Self { table, system }
    }

    fn write<C: Tuple>(&mut self, comps: C, sparse_index: Option<SparseIndex>) {
        comps.insert(self.table, sparse_index).unwrap();
    }

    fn remove<C: Tuple>(&mut self, sparse_index: SparseIndex) -> Result<C, &'static str> {
        C::remove(self.table, sparse_index)
    }

    fn query<T: Tuple>(&mut self) -> IterMut<T> {
        self.table.query(&self.system.filter)
    }
}
//-----------------SCHEDULER-----------------//
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
enum ExecutionFrequency {
    Always,
    Once,
    // Timed(f64, f64),
}

struct System {
    order: usize,
    frequency: ExecutionFrequency,
    func: fn(Command),
    run_times: usize,
    filter: Filter,
}
impl System {
    fn run(&mut self, table: &mut Table) {
        (self.func)(Command::new(table, self));
        self.run_times += 1;
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
    queue: Vec<System>,
}
impl Scheduler {
    fn new() -> Self {
        Self {
            new_pool: vec![],
            queue: vec![],
        }
    }

    fn prepare_queue(&mut self) {
        self.queue.retain(|x| match x.frequency {
            ExecutionFrequency::Always => return true,
            ExecutionFrequency::Once => {
                if x.run_times > 0 {
                    return false;
                } else {
                    return true;
                }
            }
        });
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

    fn add_system(&mut self, func: fn(Command), order: usize, once: bool, filter: Filter) {
        self.scheduler.queue.push(System {
            order,
            frequency: match once {
                true => ExecutionFrequency::Once,
                false => ExecutionFrequency::Always,
            },
            func,
            run_times: 0,
            filter,
        })
    }

    fn tick(&mut self) {
        self.scheduler.prepare_queue();
        for system in &mut self.scheduler.queue {
            system.run(&mut self.table);
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[derive(Clone)]
    struct Health(i32);
    const HEALTH: Filter = Filter::from::<Health>();

    #[derive(Clone, Debug)]
    struct Mana(i32);
    const MANA: Filter = Filter::from::<Mana>();

    #[derive(Clone, Debug)]
    struct Player(&'static str);
    const PLAYER: Filter = Filter::from::<Player>();

    #[derive(Clone, Debug)]
    struct Enemy(&'static str);
    const ENEMY: Filter = Filter::from::<Enemy>();

    fn test(mut com: Command) {
        for (player, health) in com.query::<(Player, Health)>() {
            println!("{}: {}", player.0, health.0)
        }
    }

    fn init() {
        let mut app = ECS::new();
        app.add_system(test, 0, true, PLAYER & HEALTH);
        for each in 0..10 {
            app.tick();
        }
    }

    #[test]
    fn simd() {
        let id1 = std::intrinsics::type_id::<Health>();
        let id2 = std::intrinsics::type_id::<Mana>();
        let id3 = std::intrinsics::type_id::<Player>();

        assert!(id1 & !(id2 | id3) == !(id3 | id2) & id1)
    }
}
