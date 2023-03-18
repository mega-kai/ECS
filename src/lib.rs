#![allow(dead_code, unused_variables, unused_imports, unused_mut)]
#![feature(alloc_layout_extra, map_try_insert)]
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
    pub fn yield_entity_index(&self) -> usize {
        self.entity_index
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

    pub(crate) fn contains_type(&self, comp_type: CompType) -> bool {
        let mut counter: usize = 0;
        for access in self {
            if access.column_type == comp_type {
                counter += 1;
            }
        }
        match counter {
            0 => false,
            1 => true,
            _ => panic!("contains more than one of a same type"),
        }
    }

    // for generational index
    pub(crate) fn contains_access(&self, key: TableCellAccess) -> bool {
        todo!()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.access_vec.is_empty()
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
    // usize == dense_index
    pub(crate) sparse: Vec<Option<usize>>,
}
impl TypeErasedColumn {
    // must ensure this ptr is valid first
    fn get_dense_ptr(&self, dense_index: usize) -> *mut u8 {
        unsafe {
            self.data_heap_ptr
                .add(self.comp_type.layout.size() * dense_index)
        }
    }

    pub(crate) fn yield_column_access(&self) -> AccessColumn {
        let mut result_vec = AccessColumn::new_empty(self.comp_type);
        for val in self.sparse.iter() {
            if let Some(current_id) = val {
                result_vec.0.push(TableCellAccess::new(
                    *current_id,
                    self.comp_type,
                    self.get(*current_id).unwrap(),
                ));
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

    pub(crate) fn add(&mut self, ptr: *mut u8, entity_id: usize) -> Result<*mut u8, &'static str> {
        if self.get(entity_id).is_some() {
            return Err("cell taken");
        }

        if entity_id >= self.sparse.len() {
            self.sparse.resize(entity_id + 1, None);
        }

        if self.len >= self.capacity {
            self.double_dense_cap();
        }

        self.sparse[entity_id] = Some(self.len);

        unsafe {
            let raw_dst_ptr = self.get_dense_ptr(self.len);
            std::ptr::copy(ptr, raw_dst_ptr, self.comp_type.layout.size());
            self.len += 1;
            Ok(raw_dst_ptr)
        }
    }

    pub(crate) fn replace(
        &self,
        src_ptr: *mut u8,
        entity_id: usize,
    ) -> Result<Vec<u8>, &'static str> {
        if entity_id >= self.sparse.len() {
            return Err("index overflow");
        }
        if let Some(dense_index) = self.sparse[entity_id] {
            // first allocate
            let mut vec: Vec<u8> = vec![0; self.comp_type.layout.size()];
            unsafe {
                std::ptr::copy(
                    self.get_dense_ptr(dense_index),
                    vec.as_mut_ptr(),
                    self.comp_type.layout.size(),
                );
            }
            Ok(vec)
        } else {
            Err("trying to overwrite empty cell")
        }
    }

    pub(crate) fn get(&self, entity_id: usize) -> Option<*mut u8> {
        if entity_id >= self.sparse.len() {
            None
        } else {
            if let Some(dense_index) = self.sparse[entity_id] {
                Some(self.get_dense_ptr(dense_index))
            } else {
                None
            }
        }
    }

    pub(crate) fn remove(&mut self, entity_id: usize) -> Result<*mut u8, &'static str> {
        if entity_id >= self.sparse.len() {
            Err("index overflow")
        } else {
            if let Some(dense_index) = self.sparse[entity_id] {
                self.sparse[entity_id] = None;
                Ok(self.get_dense_ptr(dense_index))
            } else {
                return Err("trying to remove empty cell");
            }
        }
    }

    // "shallow swap"
    pub(crate) fn swap(&mut self, index1: usize, index2: usize) -> Result<(), &'static str> {
        if index1 >= self.sparse.len() || index2 >= self.sparse.len() {
            Err("index overflow")
        } else {
            let dense_index1 = self.sparse[index1];
            let dense_index2 = self.sparse[index2];
            if dense_index1.is_none() || dense_index2.is_none() {
                Err("index invalid")
            } else {
                self.sparse[index1] = dense_index2;
                self.sparse[index2] = dense_index1;
                Ok(())
            }
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
    table: HashMap<CompType, TypeErasedColumn>,
    // all valid cells in this row; indexed by entity index
    row_cache: Vec<AccessRow>,
    current_entity_id: usize,
}

// TODO: cache all the comp types of all the rows, update the cache upon add/attach/remove/swap
// TODO: turning the input/output entirely on tablecell access
// TODO: incorporate all the query filter methods within the table api, making it a more proper table data structure
// TODO: variadic component insertion, probably with tuple
// TODO: generational indices
impl ComponentTable {
    pub(crate) fn new() -> Self {
        Self {
            table: HashMap::new(),
            row_cache: vec![],
            current_entity_id: 0,
        }
    }

    //-----------------COLUMN MANIPULATION-----------------//
    pub(crate) fn init_column(&mut self, comp_type: CompType) -> &mut TypeErasedColumn {
        if self.try_access(comp_type).is_ok() {
            panic!("type cannot be init twice")
        }
        self.table
            .insert(comp_type, TypeErasedColumn::new(comp_type, 64));
        self.table.get_mut(&comp_type).unwrap()
    }

    pub(crate) fn get_column(&mut self, comp_type: CompType) -> Result<AccessColumn, &'static str> {
        Ok(self.try_access(comp_type)?.yield_column_access())
    }

    pub(crate) fn pop_column(&mut self, comp_type: CompType) -> Option<TypeErasedColumn> {
        self.table.remove(&comp_type)
    }

    //-----------------ROW MANIPULATION-----------------//
    pub(crate) fn init_row(&mut self) -> usize {
        self.current_entity_id += 1;
        self.row_cache[self.current_entity_id - 1] =
            AccessRow::new_empty(self.current_entity_id - 1);
        self.current_entity_id - 1
    }

    pub(crate) fn get_row(&mut self, entity_index: usize) -> Result<AccessRow, &'static str> {
        if entity_index > self.current_entity_id {
            return Err("index overflow");
        }
        Ok(self.row_cache[entity_index].clone())
    }

    //-----------------CELL MANIPULATION HELPERS-----------------//
    fn try_access(&mut self, comp_type: CompType) -> Result<&mut TypeErasedColumn, &'static str> {
        if let Some(access) = self.table.get_mut(&comp_type) {
            Ok(access)
        } else {
            Err("no such type")
        }
    }

    fn ensure_access(&mut self, comp_type: CompType) -> &mut TypeErasedColumn {
        self.table
            .entry(comp_type)
            .or_insert(TypeErasedColumn::new(comp_type, 64))
    }

    //-----------------CELL IO-----------------//

    // if not column init, init it automatically
    pub(crate) fn push_cell(
        &mut self,
        dst_entity_index: usize,
        comp_type: CompType,
        ptr: *mut u8,
    ) -> Result<TableCellAccess, &'static str> {
        let ptr = self.ensure_access(comp_type).add(ptr, dst_entity_index)?;
        Ok(TableCellAccess::new(dst_entity_index, comp_type, ptr))
    }

    // not checking generational index
    pub(crate) fn read_cell_unchecked(
        &mut self,
        entity_index: usize,
        comp_type: CompType,
    ) -> Result<TableCellAccess, &'static str> {
        let ptr = self
            .try_access(comp_type)?
            .get(entity_index)
            .ok_or("invalid index")?;
        Ok(TableCellAccess::new(entity_index, comp_type, ptr))
    }

    pub(crate) fn pop_cell(&mut self, key: TableCellAccess) -> Result<*mut u8, &'static str> {
        self.try_access(key.column_type)?.remove(key.entity_index)
    }

    // write and return the old one in a series of bytes
    pub(crate) fn replace_cell(
        &mut self,
        key: TableCellAccess,
        ptr: *mut u8,
    ) -> Result<Vec<u8>, &'static str> {
        let access = self.try_access(key.column_type)?;
        access.replace(ptr, key.entity_index)
    }

    //-----------------CELL OPERATION WITHIN TABLE-----------------//
    // two valid cells, move one to another location, and pop that location
    pub(crate) fn replace_cell_within(
        &mut self,
        comp_type: CompType,
        from_index: usize,
        to_index: usize,
    ) -> Result<*mut u8, &'static str> {
        let access = self.try_access(comp_type)?;
        todo!()
    }

    // two valid cells
    pub(crate) fn swap_cell_within(
        &mut self,
        comp_type: CompType,
        cell1_entity_index: usize,
        cell2_entity_index: usize,
    ) -> Result<(), &'static str> {
        self.try_access(comp_type)?
            .swap(cell1_entity_index, cell2_entity_index)
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
    pub(crate) fn apply_with_filter(
        mut vec: AccessColumn,
        table: &mut ComponentTable,
    ) -> AccessColumn {
        vec.0.retain(|x| {
            if table
                .get_row(x.entity_index)
                .unwrap()
                .contains_type(FilterComp::comp_type())
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
        table: &mut ComponentTable,
    ) -> AccessColumn {
        vec.0.retain(|x| {
            if table
                .get_row(x.entity_index)
                .unwrap()
                .contains_type(FilterComp::comp_type())
            {
                return false;
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

    pub fn add_component<C: Component>(&mut self, mut component: C) -> TableCellAccess {
        let comp_type = C::comp_type();
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
        let comp_type = C::comp_type();
        // making sure they are different types
        if key.column_type == comp_type {
            return Err("type == type of access");
        }
        let row = self.table.get_row(key.entity_index)?;
        if row.contains_type(comp_type) {
            return Err("type already exists in this row");
        } else {
            let access = self.table.push_cell(
                key.entity_index,
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
        if key.column_type != C::comp_type() {
            return Err("type not matching");
        }
        let ptr = self.table.pop_cell(key)?;
        let result = unsafe { ptr.cast::<C>().as_mut().unwrap().clone() };
        Ok(result)
    }

    pub fn query<C: Component, F: Filter>(&mut self) -> AccessColumn {
        let column = self.table.get_column(C::comp_type());
        match column {
            Ok(result) => <F as Filter>::apply_on(result, self.table),
            // yield empty one
            Err(error) => AccessColumn::new_empty(C::comp_type()),
        }
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
