#![no_std]
#![no_main]

extern crate panic_halt; // Panic handler for embedded

use heapless::Vec;
use anomaly_map::{Grid2D, Cell, CellStatus};
use floodfill_core::{find_all_components, Connectivity};
use features::{FeatureTracker, extract_grid_components};
use decisions::{decide, Policy, DecisionContext, ActionType, SubsystemType};

/// Flight software configuration
const GRID_WIDTH: usize = 64;
const GRID_HEIGHT: usize = 32;
const MAX_FEATURES: usize = 16;
const MAX_COMPONENTS: usize = 32;

/// Mission time counter (milliseconds since boot)
static mut MISSION_TIME: u64 = 0;

/// Flight software state
struct FlightSoftware {
    /// Solar panel anomaly grid
    solar_grid: Grid2D<Cell>,
    /// Thermal system grid  
    thermal_grid: Grid2D<Cell>,
    /// Feature tracker for temporal analysis
    feature_tracker: FeatureTracker,
    /// Decision policy
    policy: Policy,
    /// System state
    context: DecisionContext,
    /// Performance metrics
    metrics: FlightMetrics,
}

/// Flight software performance metrics
#[derive(Clone, Copy, Debug)]
struct FlightMetrics {
    /// Total anomalies detected
    total_anomalies: u32,
    /// Actions executed
    actions_executed: u32,
    /// Average processing time in microseconds
    avg_processing_time_us: u32,
    /// Maximum stack usage
    max_stack_usage: usize,
    /// Memory usage in bytes
    memory_usage: usize,
    /// Last update timestamp
    last_update: u64,
}

impl Default for FlightMetrics {
    fn default() -> Self {
        Self {
            total_anomalies: 0,
            actions_executed: 0,
            avg_processing_time_us: 0,
            max_stack_usage: 0,
            memory_usage: 0,
            last_update: 0,
        }
    }
}

impl FlightSoftware {
    /// Initialize flight software
    pub fn new() -> Result<Self, &'static str> {
        let solar_grid = Grid2D::new(GRID_WIDTH, GRID_HEIGHT, Cell::default())
            .map_err(|_| "Failed to create solar grid")?;
            
        let thermal_grid = Grid2D::new(GRID_WIDTH, GRID_HEIGHT, Cell::default())
            .map_err(|_| "Failed to create thermal grid")?;
            
        let feature_tracker = FeatureTracker::new(MAX_FEATURES);
        let policy = Policy::default();
        let context = DecisionContext::default();
        let metrics = FlightMetrics::default();
        
        Ok(Self {
            solar_grid,
            thermal_grid,
            feature_tracker,
            policy,
            context,
            metrics,
        })
    }
    
    /// Main flight software processing cycle
    pub fn process_cycle(&mut self) -> Result<(), &'static str> {
        let start_time = get_mission_time();
        
        // Update system context from spacecraft bus
        self.update_context()?;
        
        // Process solar panel anomalies
        self.process_solar_anomalies()?;
        
        // Process thermal anomalies
        self.process_thermal_anomalies()?;
        
        // Update performance metrics
        let end_time = get_mission_time();
        let processing_time = (end_time - start_time) as u32;
        self.update_metrics(processing_time);
        
        Ok(())
    }
    
    /// Process solar panel anomaly detection
    fn process_solar_anomalies(&mut self) -> Result<(), &'static str> {
        // Simulate sensor readings (in real system, this would interface with hardware)
        self.simulate_solar_readings()?;
        
        // Extract connected components
        let components = extract_grid_components(&self.solar_grid, Connectivity::Four)
            .map_err(|_| "Failed to extract solar components")?;
        
        // Update feature tracker
        self.feature_tracker.update(components)
            .map_err(|_| "Failed to update feature tracker")?;
        
        // Make decisions based on detected features
        let features = self.feature_tracker.features();
        if !features.is_empty() {
            let action = decide(features, &self.policy, &self.context)
                .map_err(|_| "Decision engine failed")?;
                
            self.execute_action(action, SubsystemType::Power)?;
        }
        
        Ok(())
    }
    
    /// Process thermal system anomaly detection
    fn process_thermal_anomalies(&mut self) -> Result<(), &'static str> {
        // Simulate thermal sensor readings
        self.simulate_thermal_readings()?;
        
        // Extract thermal anomaly components
        let components = extract_grid_components(&self.thermal_grid, Connectivity::Eight)
            .map_err(|_| "Failed to extract thermal components")?;
        
        if !components.is_empty() {
            // For thermal anomalies, we use immediate response
            let action = if components.len() > 5 {
                // Many thermal anomalies - attitude maneuver
                ActionType::AttitudeManeuver {
                    quaternion: [0.866, 0.0, 0.5, 0.0], // 60° rotation
                    rate: 1.0,
                    hold_time: 600,
                }
            } else {
                // Few thermal anomalies - activate heaters
                ActionType::ThermalProtection {
                    heaters_on: true,
                    radiator_angle: Some(30.0),
                    louver_position: Some(0.7),
                }
            };
            
            self.execute_action(action, SubsystemType::Thermal)?;
        }
        
        Ok(())
    }
    
    /// Simulate solar panel sensor readings
    fn simulate_solar_readings(&mut self) -> Result<(), &'static str> {
        let current_time = get_mission_time();
        
        // Simulate periodic anomaly injection for testing
        if current_time % 10000 == 0 { // Every 10 seconds
            // Inject a hot spot anomaly
            if let Ok(cell) = self.solar_grid.get_mut(15, 10) {
                let _ = cell.update(450, 1200, 8, current_time); // 45°C, 120mA, 8V
            }
        }
        
        // Simulate cell aging and recovery
        for y in 0..GRID_HEIGHT {
            for x in 0..GRID_WIDTH {
                if let Ok(cell) = self.solar_grid.get_mut(x, y) {
                    // Simulate gradual cooling
                    if cell.temperature > 300 { // 30°C
                        let new_temp = cell.temperature.saturating_sub(5);
                        let _ = cell.update(new_temp, cell.current, cell.voltage, current_time);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Simulate thermal sensor readings
    fn simulate_thermal_readings(&mut self) -> Result<(), &'static str> {
        let current_time = get_mission_time();
        
        // Simulate thermal variations based on sun angle
        let sun_heating_factor = if self.context.sun_angle < 90.0 {
            1.0 + (90.0 - self.context.sun_angle) / 90.0
        } else {
            0.5 // In shadow
        };
        
        for y in 0..GRID_HEIGHT {
            for x in 0..GRID_WIDTH {
                if let Ok(cell) = self.thermal_grid.get_mut(x, y) {
                    // Base temperature varies with position and sun angle
                    let base_temp = 200 + (sun_heating_factor * 100.0) as u16; // 20-30°C range
                    
                    // Add some random variation
                    let temp_variation = (current_time as usize * (x + y)) % 50;
                    let final_temp = base_temp + temp_variation as u16;
                    
                    let _ = cell.update(final_temp, 0, 0, current_time);
                }
            }
        }
        
        Ok(())
    }
    
    /// Update spacecraft context from system bus
    fn update_context(&mut self) -> Result<(), &'static str> {
        let current_time = get_mission_time();
        
        // Simulate context updates (in real system, this would read from spacecraft bus)
        self.context.time_since_last_action = (current_time - self.metrics.last_update) as u32 / 1000;
        
        // Simulate power variations
        self.context.available_power = 80.0 + 20.0 * ((current_time as f32 / 1000.0).sin());
        
        // Simulate sun angle changes (simplified orbital mechanics)
        self.context.sun_angle = ((current_time as f32 / 5000.0) % (2.0 * 3.14159)) * 57.2958; // radians to degrees
        
        // Update system health based on anomaly count
        let anomaly_factor = (self.metrics.total_anomalies as f32 / 100.0).min(1.0);
        self.context.system_health.power = 1.0 - anomaly_factor * 0.2;
        self.context.system_health.thermal = 1.0 - anomaly_factor * 0.1;
        
        Ok(())
    }
    
    /// Execute commanded action
    fn execute_action(&mut self, action: ActionType, subsystem: SubsystemType) -> Result<(), &'static str> {
        match action {
            ActionType::NoAction => {
                // No action required
            },
            
            ActionType::Monitor { duration, rate } => {
                // In flight software, this would configure data collection
                // For now, just increment metrics
                self.metrics.actions_executed += 1;
            },
            
            ActionType::Isolate { components, duration, .. } => {
                // In real system, this would disable specific components
                // For simulation, mark components as isolated
                for &component_id in &components {
                    if component_id < (GRID_WIDTH * GRID_HEIGHT) as u32 {
                        let x = (component_id as usize) % GRID_WIDTH;
                        let y = (component_id as usize) / GRID_WIDTH;
                        
                        match subsystem {
                            SubsystemType::Power => {
                                if let Ok(cell) = self.solar_grid.get_mut(x, y) {
                                    cell.status = CellStatus::Isolated;
                                }
                            },
                            SubsystemType::Thermal => {
                                if let Ok(cell) = self.thermal_grid.get_mut(x, y) {
                                    cell.status = CellStatus::Isolated;
                                }
                            },
                            _ => {}, // Other subsystems not implemented in this example
                        }
                    }
                }
                self.metrics.actions_executed += 1;
            },
            
            ActionType::AttitudeManeuver { quaternion, rate, hold_time } => {
                // In real system, this would command attitude control
                // Update context to reflect new attitude (simplified)
                self.context.attitude = quaternion;
                self.metrics.actions_executed += 1;
            },
            
            ActionType::PowerManagement { load_shed, use_backup, .. } => {
                // Simulate power management
                if use_backup {
                    self.context.available_power *= 0.8; // Backup power is 80% of primary
                }
                self.context.available_power *= 1.0 - load_shed;
                self.metrics.actions_executed += 1;
            },
            
            ActionType::ThermalProtection { heaters_on, .. } => {
                // Simulate heater activation
                if heaters_on {
                    // Heaters consume power but increase temperature
                    self.context.available_power -= 10.0;
                    
                    // Warm up thermal grid
                    for y in 0..GRID_HEIGHT {
                        for x in 0..GRID_WIDTH {
                            if let Ok(cell) = self.thermal_grid.get_mut(x, y) {
                                let new_temp = cell.temperature.saturating_add(50); // +5°C
                                let _ = cell.update(new_temp, 0, 0, get_mission_time());
                            }
                        }
                    }
                }
                self.metrics.actions_executed += 1;
            },
            
            ActionType::Communication { emergency_beacon, .. } => {
                // In real system, this would activate communication systems
                // For simulation, just record the action
                self.metrics.actions_executed += 1;
            },
            
            ActionType::SafeMode { level, .. } => {
                // Enter safe mode - reduce all operations
                self.context.mode = decisions::SpacecraftMode::Safe;
                self.context.available_power *= 0.5; // Reduce power consumption
                self.metrics.actions_executed += 1;
            },
        }
        
        Ok(())
    }
    
    /// Update performance metrics
    fn update_metrics(&mut self, processing_time_us: u32) {
        // Update rolling average of processing time
        if self.metrics.avg_processing_time_us == 0 {
            self.metrics.avg_processing_time_us = processing_time_us;
        } else {
            // Simple exponential moving average
            self.metrics.avg_processing_time_us = 
                (self.metrics.avg_processing_time_us * 7 + processing_time_us) / 8;
        }
        
        self.metrics.last_update = get_mission_time();
        
        // Count current anomalies
        let mut anomaly_count = 0;
        for y in 0..GRID_HEIGHT {
            for x in 0..GRID_WIDTH {
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
    
    /// Get current system status
    pub fn get_status(&self) -> FlightStatus {
        FlightStatus {
            mission_time: get_mission_time(),
            metrics: self.metrics,
            active_features: self.feature_tracker.features().len(),
            available_power: self.context.available_power,
            spacecraft_mode: self.context.mode,
            sun_angle: self.context.sun_angle,
        }
    }
}

/// Flight software status report
#[derive(Clone, Copy, Debug)]
struct FlightStatus {
    mission_time: u64,
    metrics: FlightMetrics,
    active_features: usize,
    available_power: f32,
    spacecraft_mode: decisions::SpacecraftMode,
    sun_angle: f32,
}

/// Get current mission time in milliseconds
fn get_mission_time() -> u64 {
    unsafe {
        MISSION_TIME += 100; // Increment by 100ms each call
        MISSION_TIME
    }
}

/// Main entry point for flight software
#[no_mangle]
pub extern "C" fn main() -> ! {
    // Initialize flight software
    let mut flight_sw = match FlightSoftware::new() {
        Ok(sw) => sw,
        Err(_) => {
            // Initialization failed - enter safe mode
            loop {
                // Minimal safe mode operation
                unsafe { MISSION_TIME += 1000; } // 1 second increments
            }
        }
    };
    
    // Main operational loop
    loop {
        // Execute one processing cycle
        match flight_sw.process_cycle() {
            Ok(()) => {
                // Normal operation - check if we should output status
                if get_mission_time() % 5000 == 0 { // Every 5 seconds
                    let status = flight_sw.get_status();
                    // In real system, this would be telemetered to ground
                    // For now, we just update our internal state
                }
            },
            Err(_) => {
                // Processing error - enter degraded mode
                // Continue operation but reduce complexity
            }
        }
        
        // Simulate 100ms cycle time
        for _ in 0..1000 {
            // Simple delay loop (in real system, would use timer)
            unsafe { 
                core::ptr::read_volatile(&0 as *const u8);
            }
        }
    }
}

/// Panic handler - required for no_std
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // In flight software, panic should trigger safe mode
    // Reset critical systems and enter minimal operation
    loop {
        // Minimal safe operation
        unsafe {
            MISSION_TIME += 1000;
            // Flash status LED or similar
        }
    }
}
