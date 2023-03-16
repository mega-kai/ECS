#![allow(dead_code, unused_variables, unused_imports, unused_mut)]
#![feature(alloc_layout_extra)]
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Range, RangeBounds};
use std::slice::SliceIndex;
use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    collections::HashMap,
    fmt::Debug,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableCellAccess {
    // pub(crate) assigned_index: usize,
    pub(crate) entity_index: usize,
    pub(crate) column_type: CompType,
    pub(crate) access: *mut u8,
}
impl TableCellAccess {
    pub(crate) fn new(entity_id: usize, ty: CompType, access: *mut u8) -> Self {
        Self {
            entity_index: entity_id,
            column_type: ty,
            access,
        }
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
impl IntoIterator for AccessColumn {
    type Item = TableCellAccess;

    type IntoIter = std::vec::IntoIter<TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
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
pub struct AccessRow(pub(crate) Vec<TableCellAccess>, pub(crate) usize);
impl AccessRow {
    pub(crate) fn new(access_vec: Vec<TableCellAccess>, entity_id: usize) -> Self {
        Self(access_vec, entity_id)
    }

    pub(crate) fn new_empty(entity_id: usize) -> Self {
        Self(vec![], entity_id)
    }
}
impl IntoIterator for AccessRow {
    type Item = TableCellAccess;

    type IntoIter = std::vec::IntoIter<TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl<'a> IntoIterator for &'a AccessRow {
    type Item = &'a TableCellAccess;

    type IntoIter = std::slice::Iter<'a, TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl<'a> IntoIterator for &'a mut AccessRow {
    type Item = &'a mut TableCellAccess;

    type IntoIter = std::slice::IterMut<'a, TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompType {
    // for debugging
    pub(crate) type_id: TypeId,
    pub(crate) layout: Layout,
}
impl CompType {
    pub(crate) fn new<C: Component>() -> Self {
        Self {
            type_id: TypeId::of::<C>(),
            layout: Layout::new::<C>(),
        }
    }
}

pub trait Component: Clone + 'static {
    fn comp_type() -> CompType {
        CompType {
            type_id: TypeId::of::<Self>(),
            layout: Layout::new::<Self>(),
        }
    }
}

pub(crate) struct TypeErasedColumn {
    comp_type: CompType,
    data_heap_ptr: *mut u8,
    pub(crate) capacity: usize,
    pub(crate) len: usize,
    pub(crate) sparse: Vec<Option<usize>>,
}
impl TypeErasedColumn {
    pub(crate) fn query_all_dense_ptr_with_sparse_entity_id(&self) -> Vec<(usize, *mut u8)> {
        let mut result_vec: Vec<(usize, *mut u8)> = vec![];
        for val in self.sparse.iter() {
            if let Some(current_id) = val {
                result_vec.push((*current_id, unsafe { self.get(*current_id).unwrap() }));
            } else {
                continue;
            }
        }
        result_vec
    }

    pub(crate) fn new(comp_type: CompType, size: usize) -> Self {
        assert!(comp_type.layout.size() != 0, "type is a ZST",);

        let data_heap_ptr = unsafe { std::alloc::alloc(comp_type.layout.repeat(size).unwrap().0) };
        Self {
            comp_type,
            data_heap_ptr,
            capacity: size,
            len: 0,
            sparse: vec![None; size],
        }
    }

    pub(crate) fn add(&mut self, ptr: *mut u8, entity_id: usize) -> *mut u8 {
        if entity_id >= self.sparse.len() {
            self.sparse.resize(entity_id + 1, None);
        }

        if self.len >= self.capacity {
            self.double_dense_cap();
        }

        self.sparse[entity_id] = Some(self.len);

        unsafe {
            let raw_dst_ptr = self
                .data_heap_ptr
                .add(self.len * self.comp_type.layout.size());
            std::ptr::copy(ptr, raw_dst_ptr, self.comp_type.layout.size());
            raw_dst_ptr
        }
    }

    pub(crate) unsafe fn get(&self, entity_id: usize) -> Option<*mut u8> {
        if entity_id >= self.sparse.len() {
            None
        } else {
            Some(
                self.data_heap_ptr
                    .add(entity_id * self.comp_type.layout.size()),
            )
        }
    }

    pub(crate) unsafe fn remove(&mut self, entity_id: usize) -> Option<*mut u8> {
        if entity_id >= self.sparse.len() {
            None
        } else {
            let num = self.sparse[entity_id].unwrap();
            self.sparse[entity_id] = None;
            Some(self.data_heap_ptr.add(num * self.comp_type.layout.size()))
        }
    }

    fn double_dense_cap(&mut self) {
        let new_capacity = self.capacity * 2;
        let (new_layout_of_whole_vec, _) = self
            .comp_type
            .layout
            .repeat(new_capacity)
            .expect("could not repeat this layout");
        let new_data_ptr = unsafe {
            std::alloc::realloc(
                self.data_heap_ptr,
                self.comp_type
                    .layout
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

pub struct ComponentTable {
    // TODO: make this vec indexable with comptype
    table: Vec<TypeErasedColumn>,
    current_entity_id: usize,
}

// TODO: completely remove the type generics of this data strcuture
impl ComponentTable {
    pub(crate) fn new() -> Self {
        Self {
            table: vec![],
            // 0 would be reserved
            current_entity_id: 0,
        }
    }

    pub(crate) fn init_new_column(&mut self) {}

    pub(crate) fn push_row(&mut self, slice: &[u8]) {
        todo!()
    }

    pub(crate) fn try_insert_row_slice_on_index(
        &mut self,
        dst_entity_index: usize,
        slice: &[u8],
    ) -> Result<&'static str, &'static str> {
        todo!()
    }

    pub(crate) fn move_row_slice<C: Component, R: RangeBounds<CompType>>(
        &self,
        src_entity_index: usize,
        dst_entity_index: usize,
        range: R,
    ) {
        todo!()
    }

    // if range == full, mark that entity index as available
    pub(crate) fn remove_row_slice<C: Component>(&mut self) {}
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
    pub(crate) fn apply_with_filter(
        mut vec: AccessColumn,
        table: &mut ComponentTable,
    ) -> AccessColumn {
        vec.0.retain(|x| {
            for val in table[x.entity_index].clone().unwrap() {
                if val.column_type == FilterComp::comp_type() && val.entity_index != x.entity_index
                {
                    return true;
                }
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
        table: &mut ComponentTable,
    ) -> AccessColumn {
        vec.0.retain(|x| {
            for val in table[x.entity_index].clone().unwrap() {
                if val.column_type == FilterComp::comp_type() && val.entity_index != x.entity_index
                {
                    return false;
                }
            }
            return true;
        });
        vec
    }
}

pub trait Filter: Sized {
    fn apply_on(vec: AccessColumn, table: &mut ComponentTable) -> AccessColumn;
}
impl<FilterComp: Component> Filter for With<FilterComp> {
    fn apply_on(vec: AccessColumn, table: &mut ComponentTable) -> AccessColumn {
        With::<FilterComp>::apply_with_filter(vec, table)
    }
}
impl<FilterComp: Component> Filter for Without<FilterComp> {
    fn apply_on(vec: AccessColumn, table: &mut ComponentTable) -> AccessColumn {
        Without::<FilterComp>::apply_without_filter(vec, table)
    }
}
impl Filter for () {
    fn apply_on(vec: AccessColumn, table: &mut ComponentTable) -> AccessColumn {
        vec
    }
}

pub struct Command<'a> {
    table: &'a mut ComponentTable,
}
impl<'a> Command<'a> {
    pub(crate) fn new(table: &'a mut ComponentTable) -> Self {
        Self { table }
    }

    pub fn add_component<C: Component>(&mut self, component: C) -> TableCellAccess {
        // self.table.push_cell(component)
        todo!()
    }

    pub fn remove_component<C: Component>(&mut self, key: TableCellAccess) -> C {
        // self.table.remove_cell::<C>(key).unwrap()
        todo!()
    }

    pub fn query<C: Component, F: Filter>(&mut self) -> AccessColumn {
        <F as Filter>::apply_on(self.table[C::comp_type()].clone().unwrap(), self.table)
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

    pub(crate) fn run(&self, table: &mut ComponentTable) {
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
    table: ComponentTable,
    scheduler: Scheduler,
}

impl ECS {
    pub fn new() -> Self {
        Self {
            table: ComponentTable::new(),
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
        command.add_component(Player("player name"));
        println!("uwu player spawned");
    }

    fn system(mut command: Command) {
        for val in command.query::<Player, ()>().cast_vec::<Player>() {
            println!("{}", val.0);
        }
        //
    }

    fn remove(mut command: Command) {
        for key in command.query::<Player, ()>() {
            command.remove_component::<Player>(key);
            println!("component removed uwu")
        }
    }

    #[test]
    fn test() {
        let mut app = ECS::new();
        app.add_system(spawn, 0, true);
        app.add_system(system, 1, false);

        app.tick();

        app.add_system(remove, 2, true);
        app.tick();
        app.tick();
        app.tick();
        app.tick();
    }
}
