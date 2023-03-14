use crate::component::*;
use crate::storage::*;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ExecutionFrequency {
    Always,
    Once(bool),
    // Timed(f64, f64),
}

pub struct Command<'a> {
    storage: &'a mut ComponentTable,
}
impl<'a> Command<'a> {
    pub(crate) fn new(storage: &'a mut ComponentTable) -> Self {
        Self { storage }
    }

    pub fn add_component<C: Component>(&mut self, component: C) -> ComponentAccess {
        self.storage.insert(component)
    }

    pub fn remove_component<C: Component>(&mut self, key: ComponentAccess) -> C {
        self.storage.remove_as::<C>(key).unwrap()
    }

    pub fn query<C: Component, F: Filter>(&mut self) -> Vec<&mut C> {
        let access_vec = self.storage.query_single_from_type::<C>();

        todo!()
    }
}

pub struct With<C: Component> {
    phantom: PhantomData<C>,
}
impl<FilterComp: Component> With<FilterComp> {
    fn apply_to(mut vec: Vec<ComponentAccess>) -> Vec<ComponentAccess> {
        vec.retain(|x| true);
        vec
    }
}
pub struct Without<C: Component> {
    phantom: PhantomData<C>,
}
impl<FilterComp: Component> Without<FilterComp> {
    fn apply_to(mut vec: Vec<ComponentAccess>) -> Vec<ComponentAccess> {
        vec.retain(|x| true);
        vec
    }
}

pub trait Filter: Sized {
    fn apply_on(vec: Vec<ComponentAccess>) -> Vec<ComponentAccess>;
}
impl<FilterComp: Component> Filter for With<FilterComp> {
    fn apply_on(vec: Vec<ComponentAccess>) -> Vec<ComponentAccess> {
        With::<FilterComp>::apply_to(vec)
    }
}
impl<FilterComp: Component> Filter for Without<FilterComp> {
    fn apply_on(vec: Vec<ComponentAccess>) -> Vec<ComponentAccess> {
        Without::<FilterComp>::apply_to(vec)
    }
}
impl Filter for () {
    fn apply_on(vec: Vec<ComponentAccess>) -> Vec<ComponentAccess> {
        todo!()
    }
}

// impl<F0: Filter> Filter for (F0,) {}
// impl<F0: Filter, F1: Filter> Filter for (F0, F1) {}
// impl<F0: Filter, F1: Filter, F2: Filter> Filter for (F0, F1, F2) {}
// impl<F0: Filter, F1: Filter, F2: Filter, F3: Filter> Filter for (F0, F1, F2, F3) {}

pub struct System {
    order: usize,
    frequency: ExecutionFrequency,
    func: fn(Command),
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
    waiting: Vec<System>,
    pub(crate) queue: Vec<System>,
}
impl Scheduler {
    pub fn new() -> Self {
        Self {
            new_pool: vec![],
            waiting: vec![],
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
            self.queue.clear();
        }
        self.queue.sort();
    }
}
