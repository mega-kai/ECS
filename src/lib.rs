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
    const_mut_refs,
    const_type_name
)]
use std::alloc::{alloc, dealloc, realloc};
use std::hash::Hash;
use std::ops::{BitAnd, BitOr, BitXor, Not};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    collections::HashMap,
    fmt::Debug,
};

//-----------------STORAGE-----------------//
type SparseIndex = usize;
type DenseIndex = usize;

trait Component: 'static + Clone + Sized {}
impl Component for () {}

#[derive(PartialEq, Eq, Clone)]
struct AccessCell {
    ptr: *mut u8,
    comp_type: CompType,
    sparse_index: SparseIndex,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CompType {
    name: &'static str,
    type_id: TypeId,
    layout: Layout,
}
impl CompType {
    const fn new<C: 'static + Clone + Sized>() -> Self {
        Self {
            name: type_name::<C>(),
            type_id: TypeId::of::<C>(),
            layout: Layout::new::<C>(),
        }
    }
}
impl Debug for CompType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let trimmed = self.name.split("::");
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
        let comp_type = CompType::new::<C>();
        if comp_type.layout.size() == 0 {
            panic!("zst")
        }
        let ptr = unsafe { alloc(comp_type.layout.repeat(cap).unwrap().0) };
        Self {
            ptr,
            len: 0,
            cap,
            comp_type,
        }
    }

    unsafe fn resize<C: 'static + Clone + Sized>(&mut self, size: usize, default_value: C) {
        if self.len < size {
            self.ensure_cap(size + 1);
            for num in self.len..size {
                *self.ptr.add(self.comp_type.layout.size() * num).cast::<C>() =
                    default_value.clone();
            }
        }
        self.len = size;
    }

    unsafe fn ensure_cap(&mut self, cap: usize) {
        if cap == 0 {
            panic!("zero")
        }
        if cap > self.cap {
            self.ptr = realloc(
                self.ptr,
                self.comp_type.layout.repeat(self.len).unwrap().0,
                self.cap * 2 * self.comp_type.layout.size(),
            );
            self.cap *= 2;
            self.ensure_cap(cap);
        }
    }

    unsafe fn as_ref<C: 'static + Clone + Sized>(&self) -> &[C] {
        if CompType::new::<C>() == self.comp_type {
            from_raw_parts(self.ptr as *mut C, self.len)
        } else {
            panic!("type inconsistent")
        }
    }

    unsafe fn as_mut<C: 'static + Clone + Sized>(&mut self) -> &mut [C] {
        if CompType::new::<C>() == self.comp_type {
            from_raw_parts_mut(self.ptr as *mut C, self.len)
        } else {
            panic!("type inconsistent")
        }
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
    fn new_empty<C: Component>(cap: usize) -> Self {
        let result = Self {
            sparse_index_vec: TypeErasedVec::new_empty::<SparseIndex>(cap),
            comp_vec: TypeErasedVec::new_empty::<C>(cap),
        };
        result
    }

    fn push<C: Component>(&mut self, mut content: (SparseIndex, C)) -> (DenseIndex, &mut C) {
        todo!()
    }

    // with tail swap
    fn remove<C: Component>(
        &mut self,
        dense_index: DenseIndex,
    ) -> Result<(SparseIndex, C), &'static str> {
        let thing = self.as_ref::<C>();
        let result_thing = (thing.1[dense_index], thing.0[dense_index].clone());

        self.sparse_index_vec.len -= 1;
        self.comp_vec.len -= 1;

        todo!()
    }

    fn as_ref<C: 'static + Clone + Sized>(&self) -> (&[C], &[SparseIndex]) {
        unsafe { (self.comp_vec.as_ref(), self.sparse_index_vec.as_ref()) }
    }

    fn as_mut<C: 'static + Clone + Sized>(&mut self) -> (&mut [C], &mut [SparseIndex]) {
        unsafe { (self.comp_vec.as_mut(), self.sparse_index_vec.as_mut()) }
    }
}

struct SparseVec {
    dense_index_vec: TypeErasedVec,
}

impl SparseVec {
    fn new(cap: usize) -> Self {
        if cap == 0 {
            panic!("zero sized")
        }
        let mut thing = Self {
            dense_index_vec: TypeErasedVec::new_empty::<Option<DenseIndex>>(cap),
        };
        unsafe {
            thing
                .dense_index_vec
                .resize::<Option<DenseIndex>>(cap, None)
        };
        thing
    }

    fn toggle_on(
        &mut self,
        sparse_index: SparseIndex,
        dense_index_to_write: DenseIndex,
    ) -> Result<(), &'static str> {
        unsafe { self.dense_index_vec.ensure_cap(sparse_index) };
        if let None = self.as_mut()[sparse_index] {
            self.as_mut()[sparse_index] = Some(dense_index_to_write);
            Ok(())
        } else {
            Err("sparse already taken")
        }
    }

    fn toggle_off(&mut self, sparse_index: SparseIndex) -> Result<(), &'static str> {
        unsafe { self.dense_index_vec.ensure_cap(sparse_index) };
        if let Some(_) = self.as_mut()[sparse_index] {
            self.as_mut()[sparse_index] = None;
            Ok(())
        } else {
            Err("sparse not yet taken")
        }
    }

    fn as_ref(&self) -> &[Option<DenseIndex>] {
        unsafe { self.dense_index_vec.as_ref() }
    }

    fn as_mut(&mut self) -> &mut [Option<DenseIndex>] {
        unsafe { self.dense_index_vec.as_mut() }
    }
}

struct SparseSet {
    comp_type: CompType,
    dense_vec: DenseVec,
    sparse_vec: SparseVec,
}

impl SparseSet {
    fn new<C: Component>(cap: usize) -> Self {
        Self {
            comp_type: CompType::new::<C>(),
            dense_vec: DenseVec::new_empty::<C>(cap),
            sparse_vec: SparseVec::new(cap),
        }
    }

    fn insert<C: Component>(
        &mut self,
        sparse_index: SparseIndex,
        value: C,
    ) -> Result<(), &'static str> {
        if let Some(_) = self.sparse_vec.as_ref()[sparse_index] {
            Err("sparse already taken")
        } else {
            todo!()
        }
    }

    fn remove<C: Component>(&mut self, sparse_index: SparseIndex) -> Result<C, &'static str> {
        if let Some(dense_index) = self.sparse_vec.as_ref()[sparse_index] {
            self.sparse_vec.toggle_off(sparse_index)?;
            Ok(self.dense_vec.remove(dense_index)?.1)
            // TODO handle the tail swap
        } else {
            Err("invalid sparse")
        }
    }
}

struct Table {
    table: HashMap<CompType, SparseSet>,
    bottom_sparse_index: SparseIndex,
}

impl Table {
    fn new() -> Self {
        Self {
            table: HashMap::new(),
            bottom_sparse_index: 0,
        }
    }

    //-----------------COLUMN MANIPULATION-----------------//
    fn ensure_column<C: Component>(&mut self, comp_type: CompType) -> &mut SparseSet {
        self.table
            .entry(comp_type)
            .or_insert(SparseSet::new::<C>(64))
    }

    fn remove_column<C: Component>(&mut self) -> Option<SparseSet> {
        self.table.remove(&CompType::new::<C>())
    }

    fn try_column<C: Component>(&mut self) -> Result<&mut SparseSet, &'static str> {
        if let Some(access) = self.table.get_mut(&CompType::new::<C>()) {
            Ok(access)
        } else {
            Err("no such type/column")
        }
    }

    //-----------------ROW MANIPULATION-----------------//
    fn new_row(&mut self) -> SparseIndex {
        let result = self.bottom_sparse_index;
        self.bottom_sparse_index += 1;
        result
    }

    // for rn it only extends the row, will not deallocate any row

    //-----------------BASIC OPERATIONS-----------------//
    fn insert<C: Component>(
        &mut self,
        sparse_index: SparseIndex,
        value: C,
    ) -> Result<(), &'static str> {
        self.try_column::<C>()?.insert(sparse_index, value)
    }

    fn remove<C: Component>(&mut self, sparse_index: SparseIndex) -> Result<C, &'static str> {
        self.try_column::<C>()?.remove(sparse_index)
    }

    //-----------------COLUMN YIELD-----------------//
    fn as_ref<C: Component>(&mut self) -> Result<(&[C], &[SparseIndex]), &'static str> {
        Ok(self.try_column::<C>()?.dense_vec.as_ref())
    }

    fn as_mut<C: Component>(&mut self) -> Result<(&mut [C], &mut [SparseIndex]), &'static str> {
        Ok(self.try_column::<C>()?.dense_vec.as_mut())
    }
}

trait ComponentBundle: Sized {
    fn insert(self, table: &mut Table, sparse_index: SparseIndex) -> Result<(), &'static str>;
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str>;
}

impl<C0> ComponentBundle for C0
where
    C0: Component,
{
    fn insert(mut self, table: &mut Table, sparse_index: SparseIndex) -> Result<(), &'static str> {
        let thing = table.insert(sparse_index, self)?;
        Ok(())
    }
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str> {
        table.remove::<C0>(sparse_index)
    }
}

impl<C0> ComponentBundle for (C0,)
where
    C0: Component,
{
    fn insert(mut self, table: &mut Table, sparse_index: SparseIndex) -> Result<(), &'static str> {
        table.insert(sparse_index, self.0)?;
        Ok(())
    }
    fn remove(table: &mut Table, sparse_index: SparseIndex) -> Result<Self, &'static str> {
        Ok((table.remove::<C0>(sparse_index)?,))
    }
}
impl<C0, C1> ComponentBundle for (C0, C1)
where
    C0: Component,
    C1: Component,
{
    fn insert(mut self, table: &mut Table, sparse_index: SparseIndex) -> Result<(), &'static str> {
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
impl<C0, C1, C2> ComponentBundle for (C0, C1, C2)
where
    C0: Component,
    C1: Component,
    C2: Component,
{
    fn insert(mut self, table: &mut Table, sparse_index: SparseIndex) -> Result<(), &'static str> {
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
impl<C0, C1, C2, C3> ComponentBundle for (C0, C1, C2, C3)
where
    C0: Component,
    C1: Component,
    C2: Component,
    C3: Component,
{
    fn insert(mut self, table: &mut Table, sparse_index: SparseIndex) -> Result<(), &'static str> {
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

struct Command<'a> {
    table: &'a mut Table,
    system: &'a mut System,
}
impl<'a> Command<'a> {
    fn new(table: &'a mut Table, system: &'a mut System) -> Self {
        Self { table, system }
    }

    fn new_row<C: ComponentBundle>(&mut self, comps: C) {
        let sparse = self.table.new_row();
        comps.insert(self.table, sparse).unwrap();
    }

    fn write<C: ComponentBundle>(&mut self, comps: C, sparse_index: SparseIndex) {
        comps.insert(self.table, sparse_index).unwrap();
    }

    fn remove<C: ComponentBundle>(&mut self, sparse_index: SparseIndex) -> Result<C, &'static str> {
        C::remove(self.table, sparse_index)
    }

    // this does not support bundle searching yet
    fn query<C: Component>(&mut self, filter: ExpressionTree) -> Vec<&mut C> {
        // use system to cache actually
        let sparse = filter.filter::<C>(self.table);
        todo!()
    }

    // fn query_sparse<C: Component>(&mut self, filter: ExpressionTree) -> Ve {
    //     todo!()
    // }
}

#[derive(Clone)]
enum ExpressionTree {
    Single(CompType, bool),
    Compound {
        complement_flag: bool,
        head: Box<ExpressionTree>,
        op: Operation,
        tail: Box<ExpressionTree>,
    },
}
impl ExpressionTree {
    const fn new<T: Component>() -> Self {
        // this is only for comp type
        Self::Single(CompType::new::<T>(), false)
    }

    fn toggle_flag(&mut self) {
        match self {
            ExpressionTree::Single(_, val) => *val = !*val,
            ExpressionTree::Compound {
                complement_flag: val,
                ..
            } => *val = !*val,
        }
    }

    fn recursively_fmt(&self) -> String {
        let mut result = String::new();
        match self {
            Self::Single(arg0, arg1) => {
                if *arg1 {
                    result.push_str(&format!("!{:?}", arg0));
                    result
                } else {
                    result.push_str(&format!("{:?}", arg0));
                    result
                }
            }
            Self::Compound {
                complement_flag,
                head,
                op,
                tail,
            } => {
                if *complement_flag {
                    result.push('!');
                }
                result.push('(');
                result.push_str(&head.recursively_fmt());
                result.push(' ');
                result.push_str(&format!("{:?}", op));
                result.push(' ');
                result.push_str(&tail.recursively_fmt());
                result.push(')');
                result
            }
        }
    }

    fn filter<C: Component>(mut self, table: &mut Table) -> Vec<SparseIndex> {
        // need to manually get rid of the complement flags

        todo!()
    }
}
impl Debug for ExpressionTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.recursively_fmt())
    }
}

#[derive(Clone)]
enum Operation {
    And,
    Xor,
    Or,
}
impl Debug for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::And => write!(f, "&"),
            Self::Xor => write!(f, "^"),
            Self::Or => write!(f, "|"),
        }
    }
}
impl Operation {
    fn apply(&self, vec1: &[AccessCell], vec2: &[AccessCell]) -> Vec<AccessCell> {
        // match self {
        //     Operation::And => {
        //         let mut result = intersection(vec1, vec2);
        //         result
        //     }
        //     Operation::Xor => {
        //         let mut result = union_minus_intersection(vec1, vec2);
        //         result
        //     }
        //     Operation::Or => {
        //         let mut result = union(vec1, vec2);
        //         result
        //     }
        // }
        todo!()
    }
}

//-----------------QUERY BASICS-----------------//

// assuming that all slices of refs are the sparse of a single component; these slices are not sorted tho

// A & B; AND
fn intersection(vec1: &mut [SparseIndex], vec2: &mut [SparseIndex]) -> Vec<SparseIndex> {
    // the complement flag is acually taken into consideration from here
    todo!("remember iterate with self with the sparse index, make sure the row index matches before comparing")
}

// A | B; OR
fn union(vec1: &[SparseIndex], vec2: &[SparseIndex]) -> Vec<SparseIndex> {
    todo!()
}

// (A | B) & (!A | !B); XOR
fn union_minus_intersection(vec1: &[SparseIndex], vec2: &[SparseIndex]) -> Vec<SparseIndex> {
    todo!()
}

// !A; NOT
fn complement(vec1: &[SparseIndex], vec2: &[SparseIndex]) -> Vec<SparseIndex> {
    todo!()
}

// gist of things: negation first; then applying left to right
impl BitAnd for ExpressionTree {
    type Output = Self;
    fn bitand(mut self, mut rhs: Self) -> Self::Output {
        ExpressionTree::Compound {
            head: Box::new(self),
            op: Operation::And,
            tail: Box::new(rhs),
            complement_flag: false,
        }
    }
}
impl BitOr for ExpressionTree {
    type Output = Self;
    fn bitor(mut self, mut rhs: Self) -> Self::Output {
        ExpressionTree::Compound {
            head: Box::new(self),
            op: Operation::Or,
            tail: Box::new(rhs),
            complement_flag: false,
        }
    }
}
impl BitXor for ExpressionTree {
    type Output = Self;
    fn bitxor(mut self, mut rhs: Self) -> Self::Output {
        ExpressionTree::Compound {
            head: Box::new(self),
            op: Operation::Xor,
            tail: Box::new(rhs),
            complement_flag: false,
        }
    }
}

impl Not for ExpressionTree {
    type Output = Self;
    fn not(mut self) -> Self::Output {
        self.toggle_flag();
        self
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
    // cache: Vec<>,
}
impl System {
    fn default(func: fn(Command)) -> Self {
        Self {
            order: 0,
            frequency: ExecutionFrequency::Always,
            func,
            run_times: 0,
        }
    }

    fn new(order: usize, frequency: ExecutionFrequency, func: fn(Command)) -> Self {
        Self {
            order,
            frequency,
            func,
            run_times: 0,
        }
    }

    fn run(&mut self, table: &mut Table) {
        (self.func)(Command::new(table, self))
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

    fn add_system(&mut self, system: System) {
        self.new_pool.push(system);
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

    fn add_system(&mut self, func: fn(Command), order: usize, once: bool) {
        match once {
            true => self.scheduler.add_system(System {
                order,
                frequency: ExecutionFrequency::Once,
                func,
                run_times: 0,
            }),
            false => self.scheduler.add_system(System {
                order,
                frequency: ExecutionFrequency::Always,
                func,
                run_times: 0,
            }),
        }
    }

    fn tick(&mut self) {
        self.scheduler.prepare_queue();
        for system in &mut self.scheduler.queue {
            system.run(&mut self.table);
            system.run_times += 1;
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[derive(Clone)]
    struct Health(i32);
    impl Component for Health {}
    const HEALTH: ExpressionTree = ExpressionTree::new::<Health>();

    #[derive(Clone, Debug)]
    struct Mana(i32);
    impl Component for Mana {}
    const MANA: ExpressionTree = ExpressionTree::new::<Mana>();

    #[derive(Clone, Debug)]
    struct Player(&'static str);
    impl Component for Player {}
    const PLAYER: ExpressionTree = ExpressionTree::new::<Player>();

    #[derive(Clone, Debug)]
    struct Enemy(&'static str);
    impl Component for Enemy {}
    const ENEMY: ExpressionTree = ExpressionTree::new::<Enemy>();

    const NULL: ExpressionTree = ExpressionTree::new::<()>();

    fn test(mut com: Command) {
        for ent in com.query::<Player>(!ENEMY) {
            println!("{}", ent.0)
        }
    }

    #[derive(Clone)]
    struct AObj(i32);
    impl Component for AObj {}
    const A: ExpressionTree = ExpressionTree::new::<AObj>();

    #[derive(Clone, Debug)]
    struct BObj(i32);
    impl Component for BObj {}
    const B: ExpressionTree = ExpressionTree::new::<BObj>();

    #[derive(Clone, Debug)]
    struct CObj(&'static str);
    impl Component for CObj {}
    const C: ExpressionTree = ExpressionTree::new::<CObj>();

    #[derive(Clone, Debug)]
    struct DObj(&'static str);
    impl Component for DObj {}
    const D: ExpressionTree = ExpressionTree::new::<DObj>();

    #[derive(Clone, Debug)]
    struct EObj(&'static str);
    impl Component for EObj {}
    const E: ExpressionTree = ExpressionTree::new::<EObj>();

    #[derive(Clone, Debug)]
    struct FObj(&'static str);
    impl Component for FObj {}
    const F: ExpressionTree = ExpressionTree::new::<FObj>();

    #[derive(Clone, Debug)]
    struct GObj(&'static str);
    impl Component for GObj {}
    const G: ExpressionTree = ExpressionTree::new::<GObj>();

    #[test]
    fn test_linked_ops() {
        let value = A | !(B ^ !C) & (!D ^ !(E | !F));
        println!("{:?}", value);
    }
}
