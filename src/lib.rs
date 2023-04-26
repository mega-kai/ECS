#![allow(
    dead_code,
    unused_variables,
    unreachable_code,
    unused_mut,
    unused_assignments
)]
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
use std::{alloc::Layout, fmt::Debug};

//-----------------STORAGE-----------------//
type SparseIndex = usize;
type DenseIndex = usize;
type TypeId = u64;

const NULL: Filter = Filter::from::<()>();

#[cfg(target_pointer_width = "64")]
const MASK: Simd<usize, 64> = Simd::from_array([9223372036854775808; 64]);
#[cfg(target_pointer_width = "32")]
const MASK: Simd<usize, 64> = Simd::from_array([2147483648; 64]);

trait Tuple: Sized + 'static {
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

struct TypeErasedVec {
    ptr: *mut u8,
    len: usize,
    cap: usize,
    layout: Layout,
}
impl TypeErasedVec {
    fn new_empty<C: 'static + Clone + Sized>(cap: usize) -> Self {
        assert!(cap != 0, "zero capacity is not allowed");
        let layout = Layout::new::<C>();
        assert!(layout.size() != 0, "zst");
        let ptr = unsafe { alloc(layout.repeat(cap).unwrap().0) };
        Self {
            ptr,
            len: 0,
            cap,
            layout,
        }
    }

    fn push_many<C: 'static + Clone + Sized>(&mut self, grow_len: usize, value: C) {
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
                    self.layout.repeat(self.len).unwrap().0,
                    self.cap * 2 * self.layout.size(),
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
            dealloc(self.ptr, self.layout.repeat(self.cap).unwrap().0);
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
        self.comp_vec.push_many(1, content.1);
        self.sparse_index_vec.push_many(1, content.0);
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

// TODO make this indexable with sparse index -> Option<actual value>
struct SparseSet {
    sparse_vec: TypeErasedVec,
    dense_vec: DenseVec,
}

impl SparseSet {
    fn new<C: 'static + Clone + Sized>() -> Self {
        let mut thing = Self {
            sparse_vec: TypeErasedVec::new_empty::<DenseIndex>(64),
            dense_vec: DenseVec::new_empty::<C>(),
        };
        thing.sparse_vec.push_many::<DenseIndex>(64, 0);
        thing
    }

    fn insert<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: SparseIndex,
        value: C,
    ) -> Result<(), &'static str> {
        if self.as_slice()[sparse_index] == 0 {
            let dense = self.dense_vec.push((sparse_index, value));
            self.as_slice()[sparse_index] = dense;
            Ok(())
        } else {
            Err("sparse taken")
        }
    }

    // handle tail swap
    fn remove<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: SparseIndex,
    ) -> Result<C, &'static str> {
        let dense_index = self.as_slice()[sparse_index];
        if dense_index != 0 {
            self.as_slice()[sparse_index] = 0;
            let val = self.dense_vec.remove::<C>(dense_index)?.1;
            self.as_slice()[self.dense_vec.as_slice()[dense_index]] = dense_index;
            Ok(val)
        } else {
            Err("sparse empty")
        }
    }

    fn as_slice(&self) -> &mut [DenseIndex] {
        self.sparse_vec.as_slice()
    }

    // first bit is a boolean flag indicates wheather or not this slot is taken, then the rest bits are actual dense index
    fn as_simd(&self) -> &mut [Simd<DenseIndex, 64>] {
        unsafe {
            from_raw_parts_mut(
                self.sparse_vec.ptr as *mut Simd<DenseIndex, 64>,
                self.sparse_vec.len / 64,
            )
        }
    }
}

struct Table {
    table: HashMap<TypeId, SparseSet>,
    bottom_sparse_index: SparseIndex,
    cache_table: HashMap<Filter, QueryCache>,
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
        self.table
            .entry(std::intrinsics::type_id::<C>())
            .or_insert(SparseSet::new::<C>())
            .insert(sparse, value)
        // todo, update cache; when a component is added to this row, how should we determine whether of not this row belongs
        // in certain filter group;
        // generate a filter type from the newly added row, iterate over the cache table, compare this new filter type with the existing ones
        // if comply, check if reference to this row already exists, if so leave it be, if not, insert one
    }

    fn remove<C: 'static + Clone + Sized>(
        &mut self,
        sparse_index: SparseIndex,
    ) -> Result<C, &'static str> {
        self.table
            .get_mut(&std::intrinsics::type_id::<C>())
            .ok_or("comp type not in table")?
            .remove(sparse_index)
        // todo, update cache
    }

    fn raw_query(&self, filter: &Filter) -> QueryCache {
        let num = filter.get_num();
        let mut result = self.table.get(&filter.ids[1]).unwrap().as_simd().as_ref();
        if num > 1 {
            for index in 1..=num {
                let id = filter.ids[index];
                let intermediate = self.table.get(&id).unwrap().as_simd().as_ref();
                if filter.get_first_invert_flag(index) {
                    // invert the result so far
                }
                if filter.get_second_invert_flag(index) {
                    // invert the intermediate
                }
                match filter.get_operation(index) {
                    Operation::And => todo!(),
                    Operation::Or => todo!(),
                    Operation::Xor => todo!(),
                }
            }
        }
        todo!()
    }

    fn query<T: Tuple>(&mut self, filter: &Filter) -> IterMut<T> {
        if filter.is_null() {
            return IterMut::new_empty();
        }
        let cache = if let Some(val) = self.cache_table.get(filter) {
            val
        } else {
            self.cache_table
                .insert(filter.clone(), self.raw_query(filter));
            self.cache_table.get(filter).unwrap()
        };
        IterMut::new_from(cache)
    }
}

// a vector of sparse indices
struct QueryCache(TypeErasedVec);

enum Operation {
    And,
    Or,
    Xor,
}

struct IterMut<'a, T: Tuple>(&'a mut T);
impl<'a, T: Tuple> IterMut<'a, T> {
    fn new_empty() -> Self {
        todo!()
    }

    fn new_from(cache: &QueryCache) -> Self {
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
#[derive(Eq, Clone, Copy)]
struct Filter {
    // first element is the total num of comptypes in this array;
    // two negation flags, first for all the previous, second for self and an operation
    ops_flags: [u8; 9],
    // first being the final unique id after all the bitwise ops
    ids: [TypeId; 9],
    #[cfg(debug_assertions)]
    names: [&'static str; 8],
}
impl Hash for Filter {
    fn hash<H: ~const std::hash::Hasher>(&self, state: &mut H) {
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
            #[cfg(debug_assertions)]
            names: [&""; 8],
        }
    }

    fn get_num(&self) -> usize {
        todo!()
    }

    fn is_null(&self) -> bool {
        todo!()
    }

    fn get_first_invert_flag(&self, index: usize) -> bool {
        todo!()
    }

    fn get_second_invert_flag(&self, index: usize) -> bool {
        todo!()
    }

    fn get_operation(&self, index: usize) -> Operation {
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
        let player_filter = PLAYER & HEALTH & MANA & !ENEMY;
        app.add_system(test, 0, true, player_filter);
        for each in 0..10 {
            app.tick();
        }
    }

    #[test]
    fn simd() {
        let id1 = std::intrinsics::type_id::<Health>();
        let id2 = std::intrinsics::type_id::<Mana>();
        let id3 = std::intrinsics::type_id::<Player>();
        let mask = id1 & !(id2 | id3);
        assert!(mask == !id3 & (!id2 & id1));
    }
}
