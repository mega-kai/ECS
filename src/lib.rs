#![allow(dead_code, unused_variables, unused_imports, unused_mut)]
#![feature(alloc_layout_extra, map_try_insert)]
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
    use std::alloc::Layout;

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
        command
    }

    fn empty_system() {
        println!("uwu")
    }

    #[test]
    fn type_erased_vec() {
        let mut player1 = Player("player 1");
        let mut vec = TypeErasedVec::new(Layout::new::<Player>(), 16);

        // add & get
        let index1 = vec.add((&mut player1 as *mut Player) as *mut u8);
        let thing = unsafe { vec.get(index1).unwrap().cast::<Player>().as_mut().unwrap() };
        assert_eq!(thing.0, "player 1");
    }
}
