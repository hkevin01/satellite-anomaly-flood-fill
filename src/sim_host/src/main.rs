use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anomaly_map::{AnomalyGrid, CellState};
use decisions::{ActionType, DecisionContext, DecisionEngine, MissionPhase, SubsystemAction};
use features::{Component, ComponentExtractionConfig, ComponentTracker, ThreatLevel};
use floodfill_core::{flood_fill_4conn, flood_fill_8conn, FloodFillConfig};

/// Simulation configuration
const SIM_GRID_WIDTH: usize = 128;
const SIM_GRID_HEIGHT: usize = 64;
const SIM_TIME_STEP_MS: u64 = 100;
const MAX_SIM_FEATURES: usize = 50;

/// Main simulation host
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛰️  Satellite Anomaly Flood-Fill Simulation Host");
    println!("================================================");

    // Create simulation environment
    let mut sim = Simulation::new()?;

    // Run interactive simulation
    sim.run_interactive()?;

    Ok(())
}

/// Complete simulation environment
struct Simulation {
    /// Solar panel grid
    solar_grid: AnomalyGrid,
    /// Thermal radiator grid
    thermal_grid: AnomalyGrid,
    /// Star tracker sensor grid
    sensor_grid: AnomalyGrid,
    /// Component tracker
    component_tracker: ComponentTracker,
    /// Decision engine
    decision_engine: DecisionEngine,
    /// Spacecraft context
    context: DecisionContext,
    /// Simulation time (seconds since start)
    sim_time: f64,
    /// Performance metrics
    metrics: SimulationMetrics,
    /// Anomaly scenarios
    scenarios: Vec<AnomalyScenario>,
    /// Current scenario index
    current_scenario: usize,
}

/// Simulation performance metrics
#[derive(Clone, Debug)]
struct SimulationMetrics {
    /// Total processing cycles
    total_cycles: u64,
    /// Total anomalies detected
    total_anomalies: u64,
    /// Total actions executed
    total_actions: u64,
    /// Average cycle time in microseconds
    avg_cycle_time_us: u64,
    /// Peak memory usage
    peak_memory_mb: f64,
    /// Component detection metrics
    component_metrics: HashMap<String, ComponentMetrics>,
    /// Decision timing metrics
    decision_metrics: DecisionMetrics,
}

/// Component detection metrics
#[derive(Clone, Debug, Default)]
struct ComponentMetrics {
    /// Components detected
    detected: u64,
    /// Average component size
    avg_size: f64,
    /// Largest component
    max_size: usize,
    /// Processing time
    total_processing_time_us: u64,
}

/// Decision engine metrics
#[derive(Clone, Debug, Default)]
struct DecisionMetrics {
    /// Decisions made
    decisions_made: u64,
    /// Decision breakdown by action type
    action_counts: HashMap<String, u32>,
    /// Average decision time
    avg_decision_time_us: u64,
    /// Emergency actions
    emergency_actions: u32,
}

/// Anomaly scenario for testing
#[derive(Clone, Debug)]
struct AnomalyScenario {
    /// Scenario name
    name: String,
    /// Description
    description: String,
    /// Duration in seconds
    duration: f64,
    /// Anomaly injection function
    injection_type: AnomalyType,
    /// Expected threat level
    expected_threat: ThreatLevel,
    /// Expected action
    expected_action: String,
}

/// Types of anomaly injection
#[derive(Clone, Debug, PartialEq)]
enum AnomalyType {
    /// Single hot spot
    HotSpot { x: usize, y: usize, intensity: u16 },
    /// Growing circular anomaly
    Growing {
        center_x: usize,
        center_y: usize,
        max_radius: usize,
    },
    /// Linear propagation
    Linear {
        start_x: usize,
        start_y: usize,
        direction: (i32, i32),
        length: usize,
    },
    /// Random scattered anomalies
    Scattered {
        count: usize,
        intensity_range: (u16, u16),
    },
    /// Cascading failure simulation
    Cascade {
        initial_points: Vec<(usize, usize)>,
        spread_rate: f64,
    },
}

impl Default for SimulationMetrics {
    fn default() -> Self {
        Self {
            total_cycles: 0,
            total_anomalies: 0,
            total_actions: 0,
            avg_cycle_time_us: 0,
            peak_memory_mb: 0.0,
            component_metrics: HashMap::new(),
            decision_metrics: DecisionMetrics::default(),
        }
    }
}

impl Simulation {
    /// Create new simulation
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        println!("Initializing simulation environment...");

        let solar_grid_config = anomaly_map::GridConfig {
            width: SIM_GRID_WIDTH,
            height: SIM_GRID_HEIGHT,
            max_cell_age_ms: 5000,
            default_confidence: 255,
        };
        let thermal_grid_config = anomaly_map::GridConfig {
            width: SIM_GRID_WIDTH,
            height: SIM_GRID_HEIGHT,
            max_cell_age_ms: 5000,
            default_confidence: 255,
        };
        let sensor_grid_config = anomaly_map::GridConfig {
            width: 64,
            height: 64,
            max_cell_age_ms: 5000,
            default_confidence: 255,
        };

        let solar_grid = AnomalyGrid::new(solar_grid_config)?;
        let thermal_grid = AnomalyGrid::new(thermal_grid_config)?;
        let sensor_grid = AnomalyGrid::new(sensor_grid_config)?;

        let config = ComponentExtractionConfig {
            max_components: MAX_SIM_FEATURES,
            min_component_size: 3,
            max_component_size: 1000,
            iou_threshold: 0.5,
            max_age_frames: 10,
            timeout_us: 10000,
            flood_fill_config: FloodFillConfig::default(),
        };
        let component_tracker = ComponentTracker::new(config);
        let decision_engine = DecisionEngine::default();
        let context = DecisionContext {
            timestamp: 0,
            power_level: 100,
            emergency_enabled: false,
            available_power_w: 1000.0,
            attitude_quaternion: [1.0, 0.0, 0.0, 0.0],
            mission_phase: MissionPhase::Operations,
        };

        // Create test scenarios
        let scenarios = vec![
            AnomalyScenario {
                name: "Solar Panel Hot Spot".to_string(),
                description: "Single hot spot on solar panel".to_string(),
                duration: 30.0,
                injection_type: AnomalyType::HotSpot {
                    x: 32,
                    y: 16,
                    intensity: 500,
                },
                expected_threat: ThreatLevel::Low,
                expected_action: "Monitor".to_string(),
            },
            AnomalyScenario {
                name: "Growing Thermal Anomaly".to_string(),
                description: "Expanding thermal anomaly on radiator".to_string(),
                duration: 60.0,
                injection_type: AnomalyType::Growing {
                    center_x: 64,
                    center_y: 32,
                    max_radius: 15,
                },
                expected_threat: ThreatLevel::High,
                expected_action: "AttitudeManeuver".to_string(),
            },
            AnomalyScenario {
                name: "Cascading Power Failure".to_string(),
                description: "Multiple component failures with propagation".to_string(),
                duration: 120.0,
                injection_type: AnomalyType::Cascade {
                    initial_points: vec![(20, 10), (40, 20), (60, 30)],
                    spread_rate: 0.5,
                },
                expected_threat: ThreatLevel::Critical,
                expected_action: "SafeMode".to_string(),
            },
            AnomalyScenario {
                name: "Scattered Sensor Degradation".to_string(),
                description: "Random sensor pixel failures".to_string(),
                duration: 45.0,
                injection_type: AnomalyType::Scattered {
                    count: 25,
                    intensity_range: (200, 400),
                },
                expected_threat: ThreatLevel::Medium,
                expected_action: "Isolate".to_string(),
            },
            AnomalyScenario {
                name: "Linear Crack Propagation".to_string(),
                description: "Structural crack spreading linearly".to_string(),
                duration: 90.0,
                injection_type: AnomalyType::Linear {
                    start_x: 10,
                    start_y: 10,
                    direction: (1, 1),
                    length: 20,
                },
                expected_threat: ThreatLevel::High,
                expected_action: "ThermalProtection".to_string(),
            },
        ];

        println!(
            "✅ Simulation initialized with {} scenarios",
            scenarios.len()
        );

        Ok(Self {
            solar_grid,
            thermal_grid,
            sensor_grid,
            component_tracker,
            decision_engine,
            context,
            sim_time: 0.0,
            metrics: SimulationMetrics::default(),
            scenarios,
            current_scenario: 0,
        })
    }

    /// Run interactive simulation
    pub fn run_interactive(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            println!("\n🚀 Satellite Anomaly Simulation Menu:");
            println!("1. Run single scenario");
            println!("2. Run all scenarios");
            println!("3. Run custom scenario");
            println!("4. Performance benchmark");
            println!("5. Real-time monitoring");
            println!("6. View metrics");
            println!("7. Exit");

            print!("Enter choice (1-7): ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            match input.trim() {
                "1" => self.run_single_scenario()?,
                "2" => self.run_all_scenarios()?,
                "3" => self.run_custom_scenario()?,
                "4" => self.run_benchmark()?,
                "5" => self.run_realtime_monitoring()?,
                "6" => self.display_metrics(),
                "7" => {
                    println!("Simulation terminated. Final metrics:");
                    self.display_detailed_metrics();
                    break;
                }
                _ => println!("Invalid choice. Please enter 1-7."),
            }
        }

        Ok(())
    }

    /// Run a single scenario
    fn run_single_scenario(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\nAvailable scenarios:");
        for (i, scenario) in self.scenarios.iter().enumerate() {
            println!("{}. {} - {}", i + 1, scenario.name, scenario.description);
        }

        print!("Select scenario (1-{}): ", self.scenarios.len());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if let Ok(choice) = input.trim().parse::<usize>() {
            if choice > 0 && choice <= self.scenarios.len() {
                self.current_scenario = choice - 1;
                self.execute_scenario()?;
            } else {
                println!("Invalid scenario number.");
            }
        } else {
            println!("Invalid input. Please enter a number.");
        }

        Ok(())
    }

    /// Run all scenarios sequentially
    fn run_all_scenarios(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n🧪 Running all {} scenarios...", self.scenarios.len());

        let start_time = Instant::now();
        let mut results = Vec::new();

        for i in 0..self.scenarios.len() {
            self.current_scenario = i;
            println!("\n--- Scenario {}: {} ---", i + 1, self.scenarios[i].name);

            let scenario_start = Instant::now();
            let result = self.execute_scenario();
            let scenario_duration = scenario_start.elapsed();

            let success = result.is_ok();
            results.push((i + 1, success, scenario_duration));

            if let Err(e) = result {
                println!("⚠️  Scenario failed: {}", e);
            }
        }

        let total_duration = start_time.elapsed();

        // Display summary
        println!("\n📊 All Scenarios Complete - Summary:");
        println!("Total time: {:.2}s", total_duration.as_secs_f64());

        let mut passed = 0;
        for (num, success, duration) in results {
            let status = if success { "✅ PASS" } else { "❌ FAIL" };
            println!(
                "Scenario {}: {} ({:.2}s)",
                num,
                status,
                duration.as_secs_f64()
            );
            if success {
                passed += 1;
            }
        }

        println!(
            "Success rate: {}/{} ({:.1}%)",
            passed,
            self.scenarios.len(),
            (passed as f64 / self.scenarios.len() as f64) * 100.0
        );

        Ok(())
    }

    /// Execute the current scenario
    fn execute_scenario(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let scenario = self.scenarios[self.current_scenario].clone();
        println!("▶️  Executing: {}", scenario.description);

        // Reset simulation state
        self.reset_grids()?;
        self.component_tracker.reset();
        self.sim_time = 0.0;

        let scenario_start = Instant::now();
        let time_step = Duration::from_millis(SIM_TIME_STEP_MS);

        // Run scenario
        while self.sim_time < scenario.duration {
            let cycle_start = Instant::now();

            // Inject anomalies based on scenario
            self.inject_anomalies(&scenario)?;

            // Update spacecraft context
            self.update_context();

            // Process anomaly detection
            self.process_anomaly_detection()?;

            // Update metrics
            let cycle_time = cycle_start.elapsed();
            self.update_cycle_metrics(cycle_time);

            // Progress indicator
            if self.metrics.total_cycles % 50 == 0 {
                let progress = (self.sim_time / scenario.duration * 100.0) as u32;
                print!(
                    "\r🔄 Progress: {}% | Anomalies: {} | Actions: {} | Time: {:.1}s",
                    progress,
                    self.metrics.total_anomalies,
                    self.metrics.total_actions,
                    self.sim_time
                );
                io::stdout().flush()?;
            }

            // Advance simulation time
            self.sim_time += time_step.as_secs_f64();
            thread::sleep(Duration::from_millis(10)); // Slow down for visualization
        }

        println!(); // New line after progress indicator

        let total_time = scenario_start.elapsed();
        println!("✅ Scenario completed in {:.2}s", total_time.as_secs_f64());

        // Validate results
        self.validate_scenario_results(&scenario)?;

        Ok(())
    }

    /// Inject anomalies based on scenario type
    fn inject_anomalies(
        &mut self,
        scenario: &AnomalyScenario,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let current_time = get_current_timestamp();

        match &scenario.injection_type {
            AnomalyType::HotSpot { x, y, intensity } => {
                if let Ok(cell) = self.solar_grid.get_mut(*x, *y) {
                    cell.update(*intensity, 1000, 5000, current_time)?;
                }
            }

            AnomalyType::Growing {
                center_x,
                center_y,
                max_radius,
            } => {
                // Calculate current radius based on simulation time
                let progress = (self.sim_time / scenario.duration).min(1.0);
                let current_radius = (*max_radius as f64 * progress) as usize;

                for dy in 0..=current_radius {
                    for dx in 0..=current_radius {
                        let distance = ((dx * dx + dy * dy) as f64).sqrt();
                        if distance <= current_radius as f64 {
                            let positions = [
                                (*center_x + dx, *center_y + dy),
                                (center_x.saturating_sub(dx), *center_y + dy),
                                (*center_x + dx, center_y.saturating_sub(dy)),
                                (center_x.saturating_sub(dx), center_y.saturating_sub(dy)),
                            ];

                            for (x, y) in positions {
                                if x < SIM_GRID_WIDTH && y < SIM_GRID_HEIGHT {
                                    if let Ok(cell) = self.thermal_grid.get_mut(x, y) {
                                        let intensity = 400 + (distance * 50.0) as u16;
                                        cell.update(intensity, 0, 0, current_time)?;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            AnomalyType::Linear {
                start_x,
                start_y,
                direction,
                length,
            } => {
                let progress = (self.sim_time / scenario.duration).min(1.0);
                let current_length = (*length as f64 * progress) as usize;

                for i in 0..=current_length {
                    let x = (*start_x as i32 + direction.0 * i as i32) as usize;
                    let y = (*start_y as i32 + direction.1 * i as i32) as usize;

                    if x < SIM_GRID_WIDTH && y < SIM_GRID_HEIGHT {
                        if let Ok(cell) = self.solar_grid.get_mut(x, y) {
                            cell.update(350 + i as u16 * 10, 800, 4500, current_time)?;
                        }
                    }
                }
            }

            AnomalyType::Scattered {
                count,
                intensity_range,
            } => {
                if self.sim_time.fract() < 0.1 {
                    // Inject new anomalies occasionally
                    for _ in 0..*count {
                        let x = (current_time as usize * 1337) % SIM_GRID_WIDTH;
                        let y = (current_time as usize * 7919) % SIM_GRID_HEIGHT;
                        let intensity = intensity_range.0
                            + ((current_time as u16 * 31)
                                % (intensity_range.1 - intensity_range.0));

                        if let Ok(cell) = self.sensor_grid.get_mut(x % 64, y % 64) {
                            cell.update(intensity, 500, 3300, current_time)?;
                        }
                    }
                }
            }

            AnomalyType::Cascade {
                initial_points,
                spread_rate,
            } => {
                // Start with initial failure points
                for &(x, y) in initial_points {
                    if let Ok(cell) = self.solar_grid.get_mut(x, y) {
                        cell.update(600, 0, 0, current_time)?; // Dead cell
                    }
                }

                // Spread failures over time
                let spread_distance = (self.sim_time * spread_rate) as usize;
                for &(cx, cy) in initial_points {
                    for dy in 0..=spread_distance {
                        for dx in 0..=spread_distance {
                            if dx + dy <= spread_distance {
                                let positions = [
                                    (cx + dx, cy + dy),
                                    (cx.saturating_sub(dx), cy + dy),
                                    (cx + dx, cy.saturating_sub(dy)),
                                    (cx.saturating_sub(dx), cy.saturating_sub(dy)),
                                ];

                                for (x, y) in positions {
                                    if x < SIM_GRID_WIDTH && y < SIM_GRID_HEIGHT {
                                        if let Ok(cell) = self.solar_grid.get_mut(x, y) {
                                            let degradation = 300 + (dx + dy) as u16 * 20;
                                            cell.update(degradation, 200, 2000, current_time)?;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Process anomaly detection on all grids
    fn process_anomaly_detection(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let detection_start = Instant::now();

        // Extract components from all grids
        let solar_components = extract_grid_components(&self.solar_grid, Connectivity::Four)?;
        let thermal_components = extract_grid_components(&self.thermal_grid, Connectivity::Eight)?;
        let sensor_components = extract_grid_components(&self.sensor_grid, Connectivity::Four)?;

        // Update component metrics
        self.update_component_metrics("solar", &solar_components);
        self.update_component_metrics("thermal", &thermal_components);
        self.update_component_metrics("sensor", &sensor_components);

        // Combine all components for feature tracking
        let mut all_components = solar_components;
        all_components.extend(thermal_components);
        all_components.extend(sensor_components);

        // Update feature tracker
        self.feature_tracker.update(all_components)?;

        // Make decisions based on detected features
        let features = self.feature_tracker.features();
        if !features.is_empty() {
            let decision_start = Instant::now();

            let action = decide(features, &self.policy, &self.context)?;
            let decision_time = decision_start.elapsed();

            // Update decision metrics
            self.update_decision_metrics(&action, decision_time);

            // Execute action (simulated)
            self.execute_simulated_action(action)?;
        }

        let total_detection_time = detection_start.elapsed();
        self.metrics
            .component_metrics
            .entry("total_detection".to_string())
            .or_default()
            .total_processing_time_us += total_detection_time.as_micros() as u64;

        Ok(())
    }

    /// Update component detection metrics
    fn update_component_metrics(
        &mut self,
        grid_name: &str,
        components: &[floodfill_core::RegionStats],
    ) {
        let entry = self
            .metrics
            .component_metrics
            .entry(grid_name.to_string())
            .or_default();

        entry.detected += components.len() as u64;

        if !components.is_empty() {
            let total_size: usize = components.iter().map(|c| c.count).sum();
            entry.avg_size = (entry.avg_size + total_size as f64 / components.len() as f64) / 2.0;
            entry.max_size = entry
                .max_size
                .max(components.iter().map(|c| c.count).max().unwrap_or(0));

            for component in components {
                entry.total_processing_time_us += component.metrics.execution_time_us;
            }
        }
    }

    /// Update decision engine metrics
    fn update_decision_metrics(&mut self, action: &ActionType, decision_time: Duration) {
        let metrics = &mut self.metrics.decision_metrics;
        metrics.decisions_made += 1;

        let action_name = match action {
            ActionType::NoAction => "NoAction",
            ActionType::Monitor { .. } => "Monitor",
            ActionType::Isolate { .. } => "Isolate",
            ActionType::AttitudeManeuver { .. } => "AttitudeManeuver",
            ActionType::PowerManagement { .. } => "PowerManagement",
            ActionType::ThermalProtection { .. } => "ThermalProtection",
            ActionType::Communication { .. } => "Communication",
            ActionType::SafeMode { .. } => {
                metrics.emergency_actions += 1;
                "SafeMode"
            }
        };

        *metrics
            .action_counts
            .entry(action_name.to_string())
            .or_insert(0) += 1;

        // Update average decision time
        let decision_time_us = decision_time.as_micros() as u64;
        if metrics.avg_decision_time_us == 0 {
            metrics.avg_decision_time_us = decision_time_us;
        } else {
            metrics.avg_decision_time_us = (metrics.avg_decision_time_us + decision_time_us) / 2;
        }
    }

    /// Execute simulated action
    fn execute_simulated_action(
        &mut self,
        action: ActionType,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match action {
            ActionType::NoAction => {}
            ActionType::Monitor { .. } => {
                // Simulated monitoring - no state change
            }
            ActionType::Isolate { .. } => {
                // Simulate component isolation by reducing power
                self.context.available_power *= 0.95;
            }
            ActionType::AttitudeManeuver { quaternion, .. } => {
                // Update attitude
                self.context.attitude = quaternion;
            }
            ActionType::PowerManagement {
                load_shed,
                use_backup,
                ..
            } => {
                if use_backup {
                    self.context.available_power *= 0.8;
                }
                self.context.available_power *= 1.0 - load_shed;
            }
            ActionType::ThermalProtection { .. } => {
                // Simulate thermal adjustment
                self.context.available_power -= 5.0;
            }
            ActionType::Communication { .. } => {
                // Simulate communication usage
                self.context.available_power -= 2.0;
            }
            ActionType::SafeMode { .. } => {
                self.context.mode = SpacecraftMode::Safe;
                self.context.available_power *= 0.5;
            }
        }

        self.metrics.total_actions += 1;
        Ok(())
    }

    /// Update spacecraft context
    fn update_context(&mut self) {
        let current_time = get_current_timestamp();

        // Simulate orbital dynamics
        let orbital_period = 5400.0; // 90 minutes in seconds
        let orbital_phase =
            (self.sim_time % orbital_period) / orbital_period * 2.0 * std::f64::consts::PI;

        // Sun angle varies with orbital position
        self.context.sun_angle = (orbital_phase.cos() * 90.0) as f32;

        // Ground contact windows
        self.context.ground_contact = orbital_phase.sin() > 0.8;

        // Power varies with sun angle and solar panel efficiency
        let base_power = 100.0;
        let solar_efficiency = (self.context.sun_angle.to_radians().cos().max(0.0)).powf(0.5);
        self.context.available_power = base_power * solar_efficiency;

        // Update time since last action
        self.context.time_since_last_action = (self.sim_time * 1000.0) as u32;
    }

    /// Update cycle performance metrics
    fn update_cycle_metrics(&mut self, cycle_time: Duration) {
        self.metrics.total_cycles += 1;

        let cycle_time_us = cycle_time.as_micros() as u64;
        if self.metrics.avg_cycle_time_us == 0 {
            self.metrics.avg_cycle_time_us = cycle_time_us;
        } else {
            self.metrics.avg_cycle_time_us =
                (self.metrics.avg_cycle_time_us * 9 + cycle_time_us) / 10;
        }

        // Update anomaly count
        let mut anomaly_count = 0;
        for y in 0..SIM_GRID_HEIGHT {
            for x in 0..SIM_GRID_WIDTH {
                if let Ok(cell) = self.solar_grid.get(x, y) {
                    if matches!(cell.status, CellStatus::Anomalous) {
                        anomaly_count += 1;
                    }
                }
                if let Ok(cell) = self.thermal_grid.get(x, y) {
                    if matches!(cell.status, CellStatus::Anomalous) {
                        anomaly_count += 1;
                    }
                }
            }
        }
        self.metrics.total_anomalies = anomaly_count;
    }

    /// Validate scenario results
    fn validate_scenario_results(
        &self,
        scenario: &AnomalyScenario,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let features = self.feature_tracker.features();

        // Check if we detected appropriate threat levels
        let max_threat = features
            .iter()
            .map(|f| f.threat_level)
            .max()
            .unwrap_or(ThreatLevel::None);

        if max_threat < scenario.expected_threat {
            println!(
                "⚠️  Warning: Expected threat level {:?}, but maximum detected was {:?}",
                scenario.expected_threat, max_threat
            );
        }

        // Check if appropriate actions were taken
        let action_taken = self
            .metrics
            .decision_metrics
            .action_counts
            .get(&scenario.expected_action)
            .unwrap_or(&0);

        if *action_taken == 0 {
            println!(
                "⚠️  Warning: Expected action '{}' was not taken",
                scenario.expected_action
            );
        }

        println!("📊 Scenario Results:");
        println!("   Features detected: {}", features.len());
        println!("   Max threat level: {:?}", max_threat);
        println!("   Actions executed: {}", self.metrics.total_actions);
        println!("   Cycle time: {:.2}μs avg", self.metrics.avg_cycle_time_us);

        Ok(())
    }

    /// Reset all grids to default state
    fn reset_grids(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for y in 0..SIM_GRID_HEIGHT {
            for x in 0..SIM_GRID_WIDTH {
                *self.solar_grid.get_mut(x, y)? = Cell::default();
                *self.thermal_grid.get_mut(x, y)? = Cell::default();
            }
        }

        for y in 0..64 {
            for x in 0..64 {
                *self.sensor_grid.get_mut(x, y)? = Cell::default();
            }
        }

        Ok(())
    }

    /// Display current metrics
    fn display_metrics(&self) {
        println!("\n📊 Current Simulation Metrics:");
        println!("   Cycles: {}", self.metrics.total_cycles);
        println!("   Anomalies: {}", self.metrics.total_anomalies);
        println!("   Actions: {}", self.metrics.total_actions);
        println!("   Avg cycle time: {:.2}μs", self.metrics.avg_cycle_time_us);
        println!(
            "   Active features: {}",
            self.feature_tracker.features().len()
        );

        if !self.metrics.decision_metrics.action_counts.is_empty() {
            println!("\n   Action breakdown:");
            for (action, count) in &self.metrics.decision_metrics.action_counts {
                println!("     {}: {}", action, count);
            }
        }
    }

    /// Display detailed metrics
    fn display_detailed_metrics(&self) {
        println!("\n📈 Detailed Performance Metrics:");
        println!("=====================================");

        println!("Overall Statistics:");
        println!("  Total simulation cycles: {}", self.metrics.total_cycles);
        println!(
            "  Total anomalies detected: {}",
            self.metrics.total_anomalies
        );
        println!("  Total actions executed: {}", self.metrics.total_actions);
        println!(
            "  Average cycle time: {:.2}μs",
            self.metrics.avg_cycle_time_us
        );

        println!("\nComponent Detection:");
        for (grid, metrics) in &self.metrics.component_metrics {
            println!("  {} grid:", grid);
            println!("    Components detected: {}", metrics.detected);
            println!("    Average size: {:.1} cells", metrics.avg_size);
            println!("    Largest component: {} cells", metrics.max_size);
            println!(
                "    Processing time: {:.2}ms",
                metrics.total_processing_time_us as f64 / 1000.0
            );
        }

        println!("\nDecision Engine:");
        println!(
            "  Decisions made: {}",
            self.metrics.decision_metrics.decisions_made
        );
        println!(
            "  Emergency actions: {}",
            self.metrics.decision_metrics.emergency_actions
        );
        println!(
            "  Average decision time: {:.2}μs",
            self.metrics.decision_metrics.avg_decision_time_us
        );

        if !self.metrics.decision_metrics.action_counts.is_empty() {
            println!("  Action distribution:");
            for (action, count) in &self.metrics.decision_metrics.action_counts {
                let percentage =
                    (*count as f64 / self.metrics.decision_metrics.decisions_made as f64) * 100.0;
                println!("    {}: {} ({:.1}%)", action, count, percentage);
            }
        }
    }

    /// Run performance benchmark
    fn run_benchmark(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n🔥 Running Performance Benchmark...");
        println!("This will stress-test the anomaly detection system.");

        // Reset state
        self.reset_grids()?;
        self.feature_tracker.clear();
        self.metrics = SimulationMetrics::default();

        // Create high-stress scenario
        let benchmark_duration = 60.0; // 1 minute
        let high_stress_scenario = AnomalyScenario {
            name: "Benchmark Stress Test".to_string(),
            description: "High-density anomaly injection for performance testing".to_string(),
            duration: benchmark_duration,
            injection_type: AnomalyType::Scattered {
                count: 100,
                intensity_range: (300, 600),
            },
            expected_threat: ThreatLevel::High,
            expected_action: "Multiple".to_string(),
        };

        let start_time = Instant::now();
        self.sim_time = 0.0;

        // Inject many anomalies quickly
        println!("Injecting high-density anomalies...");
        for i in 0..1000 {
            let x = (i * 17) % SIM_GRID_WIDTH;
            let y = (i * 31) % SIM_GRID_HEIGHT;
            let intensity = 300 + (i % 300) as u16;

            if let Ok(cell) = self.solar_grid.get_mut(x, y) {
                cell.update(intensity, 1000, 5000, get_current_timestamp())?;
            }
        }

        // Measure processing performance
        println!("Measuring detection and decision performance...");
        for cycle in 0..1000 {
            let cycle_start = Instant::now();

            self.process_anomaly_detection()?;

            let cycle_time = cycle_start.elapsed();
            self.update_cycle_metrics(cycle_time);

            if cycle % 100 == 0 {
                print!("\r🔄 Benchmark progress: {}%", cycle / 10);
                io::stdout().flush()?;
            }
        }

        let total_time = start_time.elapsed();
        println!(
            "\n✅ Benchmark completed in {:.2}s",
            total_time.as_secs_f64()
        );

        // Display benchmark results
        println!("\n🏆 Benchmark Results:");
        println!(
            "   Throughput: {:.0} cycles/sec",
            1000.0 / total_time.as_secs_f64()
        );
        println!(
            "   Average cycle time: {:.2}μs",
            self.metrics.avg_cycle_time_us
        );
        println!("   Peak anomalies: {}", self.metrics.total_anomalies);
        println!(
            "   Components detected: {}",
            self.metrics
                .component_metrics
                .values()
                .map(|m| m.detected)
                .sum::<u64>()
        );
        println!("   Actions executed: {}", self.metrics.total_actions);

        // Performance rating
        let performance_score = if self.metrics.avg_cycle_time_us < 1000 {
            "🟢 Excellent"
        } else if self.metrics.avg_cycle_time_us < 5000 {
            "🟡 Good"
        } else if self.metrics.avg_cycle_time_us < 10000 {
            "🟠 Acceptable"
        } else {
            "🔴 Needs Optimization"
        };

        println!("   Performance rating: {}", performance_score);

        Ok(())
    }

    /// Run real-time monitoring mode
    fn run_realtime_monitoring(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n📡 Real-time Monitoring Mode");
        println!("Press Ctrl+C to exit");

        self.reset_grids()?;
        self.feature_tracker.clear();
        self.sim_time = 0.0;

        // Continuous monitoring loop
        for cycle in 0.. {
            let cycle_start = Instant::now();

            // Inject random anomalies
            if cycle % 100 == 0 {
                let x = ((cycle * 17) % SIM_GRID_WIDTH as u64) as usize;
                let y = ((cycle * 31) % SIM_GRID_HEIGHT as u64) as usize;
                if let Ok(cell) = self.solar_grid.get_mut(x, y) {
                    let _ = cell.update(400, 1000, 5000, get_current_timestamp());
                }
            }

            // Process detection
            self.process_anomaly_detection()?;
            self.update_context();

            let cycle_time = cycle_start.elapsed();
            self.update_cycle_metrics(cycle_time);

            // Display status every second
            if cycle % 10 == 0 {
                let features = self.feature_tracker.features();
                let max_threat = features
                    .iter()
                    .map(|f| f.threat_level)
                    .max()
                    .unwrap_or(ThreatLevel::None);

                print!("\r🚀 Cycle: {} | Features: {} | Max Threat: {:?} | Power: {:.1}W | Cycle: {:.0}μs",
                       cycle, features.len(), max_threat,
                       self.context.available_power, self.metrics.avg_cycle_time_us);
                io::stdout().flush()?;
            }

            // Control loop timing
            thread::sleep(Duration::from_millis(100));
            self.sim_time += 0.1;
        }

        Ok(())
    }

    /// Run custom scenario (stub for user input)
    fn run_custom_scenario(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n🔧 Custom Scenario Builder");
        println!("(Simplified version - inject single hot spot)");

        print!("Enter X coordinate (0-{}): ", SIM_GRID_WIDTH - 1);
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let x: usize = input.trim().parse().unwrap_or(32);

        print!("Enter Y coordinate (0-{}): ", SIM_GRID_HEIGHT - 1);
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let y: usize = input.trim().parse().unwrap_or(16);

        print!("Enter intensity (100-1000): ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;
        let intensity: u16 = input.trim().parse().unwrap_or(400);

        let custom_scenario = AnomalyScenario {
            name: "Custom Hot Spot".to_string(),
            description: format!(
                "User-defined hot spot at ({}, {}) with intensity {}",
                x, y, intensity
            ),
            duration: 30.0,
            injection_type: AnomalyType::HotSpot { x, y, intensity },
            expected_threat: ThreatLevel::Medium,
            expected_action: "Monitor".to_string(),
        };

        // Temporarily add custom scenario
        self.scenarios.push(custom_scenario);
        self.current_scenario = self.scenarios.len() - 1;

        self.execute_scenario()?;

        // Remove custom scenario
        self.scenarios.pop();

        Ok(())
    }
}

/// Get current timestamp in milliseconds
fn get_current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
