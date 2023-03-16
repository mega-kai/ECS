#![allow(dead_code, unused_variables, unused_imports, unused_mut)]
#![feature(alloc_layout_extra)]
use std::marker::PhantomData;
use std::{
    alloc::Layout,
    any::{type_name, TypeId},
    collections::HashMap,
    fmt::Debug,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableCellAccess {
    pub(crate) row_index: usize,
    pub(crate) column_index: CompType,
    pub(crate) access: *mut u8,
}
impl TableCellAccess {
    pub(crate) fn new(entity_id: usize, ty: CompType, access: *mut u8) -> Self {
        Self {
            row_index: entity_id,
            column_index: ty,
            access,
        }
    }

    // not recommended
    pub unsafe fn cast<C: Component>(&self) -> &mut C {
        assert_eq!(C::id(), self.column_index);
        self.access.cast::<C>().as_mut().unwrap()
    }
}

// all with the same component type
pub struct AccessVec(pub(crate) Vec<TableCellAccess>);
impl AccessVec {
    pub(crate) fn new(access_vec: Vec<TableCellAccess>) -> Self {
        Self(access_vec)
    }

    pub(crate) fn new_empty() -> Self {
        Self(vec![])
    }

    pub fn cast_vec<C: Component>(&self) -> Vec<&mut C> {
        assert!(!self.0.is_empty());
        // on the promise that all accesses of this vec share the same type
        assert_eq!(C::id(), self.0[0].column_index);
        self.into_iter()
            .map(|x| unsafe { x.cast::<C>() })
            .collect::<Vec<&mut C>>()
    }
}
impl IntoIterator for AccessVec {
    type Item = TableCellAccess;

    type IntoIter = std::vec::IntoIter<TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl<'a> IntoIterator for &'a AccessVec {
    type Item = &'a TableCellAccess;

    type IntoIter = std::slice::Iter<'a, TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl<'a> IntoIterator for &'a mut AccessVec {
    type Item = &'a mut TableCellAccess;

    type IntoIter = std::slice::IterMut<'a, TableCellAccess>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompType {
    // for debugging
    pub(crate) name: &'static str,
    pub(crate) type_id: TypeId,
}
impl CompType {
    pub(crate) fn new<C: Component>() -> Self {
        Self {
            name: type_name::<C>(),
            type_id: TypeId::of::<C>(),
        }
    }
}

pub trait Component: Clone + 'static {
    fn id() -> CompType {
        CompType {
            name: type_name::<Self>(),
            type_id: TypeId::of::<Self>(),
        }
    }

    fn id_instance(&self) -> CompType {
        CompType {
            name: type_name::<Self>(),
            type_id: TypeId::of::<Self>(),
        }
    }
}

// impl<C0: Component> Component for (C0,) {}
// impl<C0: Component, C1: Component> Component for (C0, C1) {}
// impl<C0: Component, C1: Component, C2: Component> Component for (C0, C1, C2) {}
// impl<C0: Component, C1: Component, C2: Component, C3: Component> Component for (C0, C1, C2, C3) {}

pub(crate) struct TypeErasedColumn {
    layout_of_component: Layout,
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

    pub(crate) fn new(layout: Layout, size: usize) -> Self {
        assert!(layout.size() != 0, "type is a ZST",);

        let data_heap_ptr = unsafe { std::alloc::alloc(layout.repeat(size).unwrap().0) };
        Self {
            layout_of_component: layout,
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
                .add(self.len * self.layout_of_component.size());
            std::ptr::copy(ptr, raw_dst_ptr, self.layout_of_component.size());
            raw_dst_ptr
        }
    }

    pub(crate) unsafe fn get(&self, entity_id: usize) -> Option<*mut u8> {
        if entity_id >= self.sparse.len() {
            None
        } else {
            Some(
                self.data_heap_ptr
                    .add(entity_id * self.layout_of_component.size()),
            )
        }
    }

    pub(crate) unsafe fn remove(&mut self, entity_id: usize) -> Option<*mut u8> {
        if entity_id >= self.sparse.len() {
            None
        } else {
            let num = self.sparse[entity_id].unwrap();
            self.sparse[entity_id] = None;
            Some(
                self.data_heap_ptr
                    .add(num * self.layout_of_component.size()),
            )
        }
    }

    fn double_dense_cap(&mut self) {
        let new_capacity = self.capacity * 2;
        let (new_layout_of_whole_vec, _) = self
            .layout_of_component
            .repeat(new_capacity)
            .expect("could not repeat this layout");
        let new_data_ptr = unsafe {
            std::alloc::realloc(
                self.data_heap_ptr,
                self.layout_of_component
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
    data_hash: HashMap<CompType, TypeErasedColumn>,
    current_entity_id: usize,
}

impl ComponentTable {
    pub(crate) fn new() -> Self {
        Self {
            data_hash: HashMap::new(),
            // 0 would be reserved
            current_entity_id: 0,
        }
    }

    fn ensure_access_of_type<C: Component>(&mut self) -> &mut TypeErasedColumn {
        self.data_hash
            .entry(C::id())
            .or_insert(TypeErasedColumn::new(Layout::new::<C>(), 64))
    }

    fn try_access<C: Component>(&mut self) -> Result<&mut TypeErasedColumn, &'static str> {
        if let Some(access) = self.data_hash.get_mut(&C::id()) {
            Ok(access)
        } else {
            Err("no such component type exist in this storage")
        }
    }

    // should be insert multiple with [C]
    pub(crate) fn add_new_entity<C: Component>(&mut self, mut component: C) -> TableCellAccess {
        self.current_entity_id += 1;
        let new_entity_id = self.current_entity_id;
        let dst_ptr = self
            .ensure_access_of_type::<C>()
            .add((&mut component as *mut C).cast::<u8>(), new_entity_id - 1);
        TableCellAccess::new(new_entity_id - 1, CompType::new::<C>(), dst_ptr)
    }

    pub(crate) fn remove_as<C: Component>(
        &mut self,
        key: TableCellAccess,
    ) -> Result<C, &'static str> {
        unsafe {
            Ok(self
                .try_access::<C>()?
                .remove(key.row_index)
                .unwrap()
                .cast::<C>()
                .as_mut()
                .cloned()
                .unwrap())
        }
    }

    pub(crate) fn add_n_link<C: Component>(
        &mut self,
        entity_id: usize,
        comp: C,
    ) -> Result<TableCellAccess, &'static str> {
        todo!()
    }

    pub(crate) fn link_multiple(&self, entity_id: usize) {
        todo!()
    }

    pub(crate) fn query_single_from_type<C: Component>(&self) -> AccessVec {
        if let Some(access) = self.data_hash.get(&C::id()) {
            let mut raw_vec = access.query_all_dense_ptr_with_sparse_entity_id();
            let mut result_access_vec = AccessVec::new_empty();
            for (index, ptr) in raw_vec {
                result_access_vec
                    .0
                    .push(TableCellAccess::new(index, CompType::new::<C>(), ptr));
            }
            result_access_vec
        } else {
            panic!("no such component type within the table")
        }
    }

    pub(crate) fn query_accesses_with_same_id(&self, entity_id: usize) -> AccessVec {
        let mut vec = AccessVec::new_empty();
        for (id, column) in self.data_hash.iter() {
            if let Some(ptr) = unsafe { column.get(entity_id) } {
                vec.0.push(TableCellAccess::new(entity_id, *id, ptr));
            } else {
                continue;
            }
        }
        vec
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
    pub(crate) fn apply_with_filter(mut vec: AccessVec, storage: &mut ComponentTable) -> AccessVec {
        vec.0.retain(|x| {
            for val in storage.query_accesses_with_same_id(x.row_index) {
                if val.column_index == FilterComp::id() {
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
        mut vec: AccessVec,
        storage: &mut ComponentTable,
    ) -> AccessVec {
        vec.0.retain(|x| {
            for val in storage.query_accesses_with_same_id(x.row_index) {
                if val.column_index == FilterComp::id() {
                    return false;
                }
            }
            return true;
        });
        vec
    }
}

pub trait Filter: Sized {
    fn apply_on(vec: AccessVec, storage: &mut ComponentTable) -> AccessVec;
}
impl<FilterComp: Component> Filter for With<FilterComp> {
    fn apply_on(vec: AccessVec, storage: &mut ComponentTable) -> AccessVec {
        With::<FilterComp>::apply_with_filter(vec, storage)
    }
}
impl<FilterComp: Component> Filter for Without<FilterComp> {
    fn apply_on(vec: AccessVec, storage: &mut ComponentTable) -> AccessVec {
        Without::<FilterComp>::apply_without_filter(vec, storage)
    }
}
impl Filter for () {
    fn apply_on(vec: AccessVec, storage: &mut ComponentTable) -> AccessVec {
        vec
    }
}

// impl<F0: Filter> Filter for (F0,) {}
// impl<F0: Filter, F1: Filter> Filter for (F0, F1) {}
// impl<F0: Filter, F1: Filter, F2: Filter> Filter for (F0, F1, F2) {}
// impl<F0: Filter, F1: Filter, F2: Filter, F3: Filter> Filter for (F0, F1, F2, F3) {}

pub struct Command<'a> {
    storage: &'a mut ComponentTable,
}
impl<'a> Command<'a> {
    pub(crate) fn new(storage: &'a mut ComponentTable) -> Self {
        Self { storage }
    }

    pub fn add_component<C: Component>(&mut self, component: C) -> TableCellAccess {
        self.storage.add_new_entity(component)
    }

    pub fn remove_component<C: Component>(&mut self, key: TableCellAccess) -> C {
        self.storage.remove_as::<C>(key).unwrap()
    }

    pub fn query<C: Component, F: Filter>(&mut self) -> Vec<&mut C> {
        let access_vec =
            <F as Filter>::apply_on(self.storage.query_single_from_type::<C>(), self.storage);
        let mut result: Vec<&mut C> = vec![];
        for val in access_vec {
            result.push(unsafe { val.access.cast::<C>().as_mut().unwrap() });
        }
        result
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

    pub(crate) fn run(&self, storage: &mut ComponentTable) {
        (self.func)(Command::new(storage))
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
    storage: ComponentTable,
    scheduler: Scheduler,
}

impl ECS {
    pub fn new() -> Self {
        Self {
            storage: ComponentTable::new(),
            scheduler: Scheduler::new(),
        }
    }

    pub fn add_system(&mut self, system: System) {
        self.scheduler.add_system(system);
    }

    pub fn tick(&mut self) {
        self.scheduler.prepare_queue();
        for system in &mut self.scheduler.queue {
            match system.frequency {
                ExecutionFrequency::Always => system.run(&mut self.storage),
                ExecutionFrequency::Once(_) => {
                    system.frequency = ExecutionFrequency::Once(true);
                    system.run(&mut self.storage);
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
        command.add_component(Player("uwu"));
        println!("uwu player spawned");
    }
    fn system(mut command: Command) {
        for val in command.query::<Player, ()>() {
            println!("{}", val.0);
        }
        //
    }
    fn remove(mut command: Command) {
        for key in command.query::<Player, ()>() {
            // command.remove_component(key);
        }
    }

    #[test]
    fn test() {
        let mut app = ECS::new();
        app.add_system(System::new(0, ExecutionFrequency::Once(false), spawn));
        app.add_system(System::new(1, ExecutionFrequency::Always, system));

        app.tick();

        app.add_system(System::new(2, ExecutionFrequency::Once(false), remove));
        app.tick();
    }
}
