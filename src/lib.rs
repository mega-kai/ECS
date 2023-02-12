#![allow(dead_code, unused_variables, unused_imports, unused_mut)]
#![feature(alloc_layout_extra)]
mod command_buffer;
mod component;
mod query;
mod scheduler;
mod storage;
mod system;
use command_buffer::*;
use component::*;
use query::*;
use scheduler::*;
use storage::*;
use system::*;

/// the entity component system;
/// the gist of it is at any given time during any system of the same
/// execution cycle(within the same next() method) run, only one
/// mutable reference to a single component can be issued,
/// with that reference you can edit/delete that component
pub struct ECS {
    //for all components (including all entities)
    storage: Storage,
    //for all systems along with their metadata
    scheduler: Scheduler,
    //commands collected and to be executed and emptied each cycle ends
    command_buffer: CommandBuffer,
}

/// these are APIs to interact with the ECS, should be as simple as possible
impl ECS {
    /// not much to be said about this, just init new empty containers
    /// and all that
    pub fn new() -> Self {
        Self {
            storage: Storage::new(),
            scheduler: Scheduler::new(),
            command_buffer: CommandBuffer::new(),
        }
    }

    pub fn add_system<Param: SysParam + 'static, Sys: SysFn<Param> + 'static>(
        &mut self,
        func: SystemWithMetadata<Param, Sys>,
    ) {
        self.scheduler.add_system(func);
    }

    /// a tick cycle consisting of multiple stages
    pub fn tick(&mut self) {
        //generating a new queue to be executed: queue generation stages
        //this is all side effects, internally mutating the queue field
        //let order = self.scheduler.generate_queue_for_current_tick();

        //supplying with all the requested query results: query stage

        //executing all the systems in that queue sequentially: execution stage
        self.scheduler.execute_all();

        //collecting all the commands and executing them: command stage
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    struct Health(i32);
    impl Component for Health {}

    #[derive(Debug, Clone, Copy)]
    struct Mana(i32);
    impl Component for Mana {}

    #[derive(Debug, Clone, Copy)]
    struct Player(&'static str);
    impl Component for Player {}

    #[derive(Debug, Clone, Copy)]
    struct Enemy;
    impl Component for Enemy {}

    fn player_bundle() -> (Player, Health, Mana) {
        (Player("Kai"), Health(100), Mana(100))
    }

    // has a standard format of input and output
    fn spawn_player(mut command: Command) -> Command {
        println!("hello ecs!");

        let bundle = player_bundle();
        //command.spawn_bundle(bundle);

        let player = command.spawn_component(bundle.0);
        let health_comp = command.spawn_component(bundle.1);
        let mana_comp = command.spawn_component(bundle.2);
        command.link_component(player, health_comp);
        command.link_component(player, mana_comp);

        command
    }

    fn system_test(mut command: Command, mut query: Query<&mut Health, With<&Player>>) -> Command {
        todo!()
    }

    fn empty_system() {
        println!("uwu")
    }

    #[test]
    fn ecs() {
        //new ecs
        let mut ecs = ECS::new();

        //add systems
        ecs.add_system(SystemWithMetadata::once(spawn_player));

        //tick cycling
        loop {
            ecs.tick();
        }
    }

    #[test]
    fn storage() {
        let mut vec = TypeErasedVec::new::<Player>();
        let ptr0 = (&mut Player("pinita") as *mut Player).cast::<u8>();
        let ptr1 = (&mut Player("kai") as *mut Player).cast::<u8>();
        let ptr2 = (&mut Player("wolfter") as *mut Player).cast::<u8>();
        vec.push(ptr0);
        vec.push(ptr1);
        vec.push(ptr2);
        let thing0 = unsafe {
            vec.get_ptr_from_index(0)
                .unwrap()
                .cast::<Player>()
                .as_ref()
                .unwrap()
        };
        let thing1 = unsafe {
            vec.get_ptr_from_index(1)
                .unwrap()
                .cast::<Player>()
                .as_ref()
                .unwrap()
        };
        let thing2 = unsafe {
            vec.get_ptr_from_index(2)
                .unwrap()
                .cast::<Player>()
                .as_ref()
                .unwrap()
        };
        println!("len:{}, cap:{}", vec.len(), vec.cap());
        println!(
            "first: {}, second: {}, third: {}",
            thing0.0.to_uppercase(),
            thing1.0.to_uppercase(),
            thing2.0.to_uppercase()
        );
    }

    #[test]
    fn num() {}
}
