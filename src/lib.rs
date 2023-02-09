#![allow(dead_code, unused_variables)]
#![feature(alloc_layout_extra)]
mod component;
mod storage;
use component::*;
use storage::*;

/// generational indices
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
struct Key {
    index: usize,
    generation: usize,
}
impl Key {
    fn new(index: usize) -> Self {
        Self {
            index,
            generation: 0,
        }
    }
}

struct Scheduler {
    pool: Vec<fn()>,
}
impl Scheduler {
    fn new() -> Self {
        Self { pool: vec![] }
    }

    fn add_system(&mut self, func: fn()) {
        self.pool.push(func);
    }

    fn run_all(&self) {
        for system in &self.pool {
            (system)()
        }
    }
}

/// the entity component system;
/// the gist of it is at any given time during any system of the same
/// execution cycle(within the same next() method) run, only one
/// mutable reference to a single component can be issued,
/// with that reference you can edit/delete that component
struct ECS {
    storage: Storage,
    scheduler: Scheduler,
}
impl ECS {
    fn new() -> Self {
        Self {
            storage: Storage::new(),
            scheduler: Scheduler::new(),
        }
    }

    fn next(&mut self) {
        self.scheduler.run_all();
    }

    fn add_system(&mut self, func: fn()) {
        self.scheduler.add_system(func);
    }

    fn spawn<C: Component>(&mut self, component: C) -> Key {
        //generate key
        let key = Key::new(0);
        self.storage.add_component(component);
        key
    }

    fn query<C: ComponentRef>(&mut self) -> Option<C> {
        self.storage.query::<C>()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct Test(i32);
    impl Component for Test {}

    fn test() {
        println!("hello ecs!")
    }

    #[test]
    fn sparse_set() {
        let mut ecs = ECS::new();
        ecs.add_system(test);
        ecs.next();

        // what is the key used for??
        let key = ecs.spawn(Test(12));

        // can either be <&Test> <&mut Test>
        let exclusive_reference_to_test = ecs.query::<&mut Test>().unwrap();
        exclusive_reference_to_test.0 += 1;
    }
}
