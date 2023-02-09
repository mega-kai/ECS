#![allow(dead_code, unused_variables)]
#![feature(alloc_layout_extra)]
mod component;
mod scheduler;
mod storage;
use component::*;
use scheduler::*;
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

    /// return an iterator of &C/&mut C, querying a
    /// single component by an id or index is not allowed;
    ///
    /// should also be able to query for (&mut A, ..., With/Or/Without<&B>, ...)
    /// which are basically archetypes;
    ///
    /// a distinction between (&mut A, &B) and (&mut A, With<B>) is that
    /// the latter does not issue a shared reference iterator of B, thus in
    /// the same cycle an iterator of mutable refs can still be issued, as
    /// With only guarantees that such component exists along with the
    /// ones you are querying, while &B can give you read access to its data
    fn query<C: QueryIdentifier>(&mut self) -> Option<C> {
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
        //return value should be an array of references
        let exclusive_reference_to_test = ecs.query::<&mut Test>().unwrap();
        exclusive_reference_to_test.0 += 1;
    }
}
