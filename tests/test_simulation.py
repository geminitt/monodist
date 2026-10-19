from monodist.simulation import Controller, Observation, World, scenario


def observations(world):
    # Explicit ideal synthetic fixture; this test does not exercise the learned perception model.
    return [Observation(obj.id,obj.z-world.position,world.speed-obj.speed,x=obj.x,
                        width=obj.width,length=obj.length,kind=obj.kind) for obj in world.obstacles]


def test_control_changes_future_observations_and_prevents_nominal_collision():
    controlled, baseline = scenario('stationary_obstacle'), scenario('stationary_obstacle')
    policy = Controller()
    for _ in range(100):
        controlled.step(policy.decide(controlled.speed,observations(controlled)).acceleration)
        baseline.step(0.0)
    assert baseline.collision and not controlled.collision
    assert controlled.position < baseline.position
    assert observations(controlled)[0].depth > observations(baseline)[0].depth


def test_adjacent_obstacle_does_not_trigger_braking_and_late_case_can_fail():
    policy = Controller()
    adjacent = scenario('adjacent_lane')
    assert policy.decide(adjacent.speed,observations(adjacent)).mode == 'cruise'
    late = scenario('late_obstacle')
    for _ in range(30):
        late.step(policy.decide(late.speed,observations(late)).acceleration)
    assert late.collision  # the demo must retain a failure case, not promise every obstacle is avoidable


def test_hard_brake_stops_without_reversing():
    world = World(speed=1.0,dt=1.0)
    world.step(-6.0)
    assert world.speed == 0 and abs(world.position-1/12) < 1e-9
