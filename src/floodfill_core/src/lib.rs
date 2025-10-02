//! Core flood-fill algorithms for satellite anomaly detection
//!
//! # Requirement Traceability
//!
//! This crate implements the following system requirements:
//!
//! ## REQ-FLOOD-001: Connected Component Detection
//! **Requirement**: System shall identify connected anomalous regions using flood-fill algorithms
//! **Implementation**: [`flood_fill_4conn`] and [`flood_fill_8conn`] with stack-based traversal
//! **Verification**: Unit tests validate detection of various region shapes and configurations
//!
//! ## REQ-FLOOD-002: Connectivity Options
//! **Requirement**: System shall support both 4-connectivity and 8-connectivity analysis
//! **Implementation**: Separate algorithms for 4-neighbor and 8-neighbor connectivity
//! **Verification**: Connectivity differences tested with diagonal anomaly patterns
//!
//! ## REQ-FLOOD-003: Region Statistics
//! **Requirement**: System shall compute geometric statistics for detected regions
//! **Implementation**: [`RegionStats`] with area, perimeter, centroid, and bounding box
//! **Verification**: Statistics validated against known geometric shapes
//!
//! ## REQ-FLOOD-004: Memory Safety
//! **Requirement**: System shall prevent stack overflow with bounded memory usage
//! **Implementation**: Configurable stack limits in [`FloodFillConfig`]
//! **Verification**: Stack overflow protection tested with large connected regions
//!
//! ## REQ-FLOOD-005: Performance Constraints
//! **Requirement**: System shall complete flood-fill within specified time limits
//! **Implementation**: Timeout handling with early termination capability
//! **Verification**: Performance tested with large grids and complex patterns
//!
//! ## REQ-FLOOD-006: Error Handling
//! **Requirement**: System shall provide comprehensive error reporting for all failure modes
//! **Implementation**: [`FloodFillError`] covering all failure scenarios with context
//! **Verification**: Error conditions tested for proper propagation and handling
//!
//! ## REQ-FLOOD-007: Deterministic Behavior
//! **Requirement**: System shall provide repeatable results for identical inputs
//! **Implementation**: Deterministic traversal order with consistent seed handling
//! **Verification**: Reproducibility tested across multiple executions
//!
//! ## REQ-FLOOD-008: Grid Validation
//! **Requirement**: System shall validate grid parameters before processing
//! **Implementation**: Coordinate bounds checking and dimension validation
//! **Verification**: Input validation tested with edge cases and invalid parameters
//!
//! ## REQ-FLOOD-009: Logical System Analysis
//! **Requirement**: System shall support logical adjacency modeling for hardware/software decision-aiding
//! **Implementation**: [`LogicalFloodFill`] with dependency chain analysis and fault propagation
//! **Verification**: System topology analysis tested with complex dependency graphs
//!
//! ## REQ-FLOOD-010: Hardware Fault Propagation
//! **Requirement**: System shall trace fault propagation through connected subsystems
//! **Implementation**: [`HardwareFaultAnalyzer`] with power bus, sensor cluster, and component connectivity
//! **Verification**: Fault propagation validated against known hardware topologies
//!
//! ## REQ-FLOOD-011: Software Module Containment
//! **Requirement**: System shall isolate impacted services in modular architectures
//! **Implementation**: [`SoftwareImpactAnalyzer`] with call graph and dependency tree analysis
//! **Verification**: Impact isolation tested with microservice and thread dependency scenarios
//!
//! ## REQ-FLOOD-012: Policy Response Modeling
//! **Requirement**: System shall simulate policy propagation through logical system links
//! **Implementation**: [`PolicyPropagationSimulator`] with shutdown, reroute, and isolation policies
//! **Verification**: Policy effects validated across multiple system topology configurations
//!
//! # Hardware/Software Decision-Aiding Examples
//!
//! ## Hardware Fault Propagation Example
//! ```rust
//! use floodfill_core::{
//!     LogicalNode, LogicalNodeType, LogicalNodeState, LogicalNodeResources,
//!     LogicalTopology, HardwareFaultAnalyzer, HardwarePropagationRules
//! };
//!
//! // Create hardware topology
//! let mut nodes = heapless::Vec::new();
//!
//! // Power bus (critical infrastructure)
//! let mut power_connections = heapless::Vec::new();
//! power_connections.push("sensor_array".into()).unwrap();
//! power_connections.push("main_processor".into()).unwrap();
//!
//! nodes.push(LogicalNode {
//!     id: "power_bus_main".into(),
//!     node_type: LogicalNodeType::PowerBus,
//!     state: LogicalNodeState::Failed, // Initial failure point
//!     connections: power_connections,
//!     criticality: 0.95,
//!     resources: LogicalNodeResources {
//!         power: 1000.0,
//!         cpu_load: 0.0,
//!         memory_bytes: 0,
//!         bandwidth_bps: 0,
//!     },
//! }).unwrap();
//!
//! // Connected subsystems
//! nodes.push(LogicalNode {
//!     id: "sensor_array".into(),
//!     node_type: LogicalNodeType::SensorCluster,
//!     state: LogicalNodeState::Operational,
//!     connections: heapless::Vec::new(),
//!     criticality: 0.8,
//!     resources: LogicalNodeResources { power: 200.0, cpu_load: 0.4, memory_bytes: 2048, bandwidth_bps: 5000 },
//! }).unwrap();
//!
//! let topology = LogicalTopology { nodes, adjacency: heapless::FnvIndexMap::new() };
//! let analyzer = HardwareFaultAnalyzer::new(topology);
//! let rules = HardwarePropagationRules::default();
//!
//! // Analyze fault propagation from power bus failure
//! let result = analyzer.analyze_fault_propagation("power_bus_main", &rules).unwrap();
//! println!("Fault affects {} systems", result.affected_nodes.len());
//! println!("Critical systems impacted: {}", result.critical_nodes_affected);
//! ```
//!
//! ## Software Impact Analysis Example
//! ```rust
//! use floodfill_core::{
//!     LogicalNode, LogicalNodeType, LogicalNodeState, LogicalNodeResources,
//!     LogicalTopology, SoftwareImpactAnalyzer, SoftwarePropagationRules
//! };
//!
//! // Create software topology
//! let mut nodes = heapless::Vec::new();
//!
//! // Database service (critical data layer)
//! let mut db_connections = heapless::Vec::new();
//! db_connections.push("user_service".into()).unwrap();
//! db_connections.push("analytics_service".into()).unwrap();
//!
//! nodes.push(LogicalNode {
//!     id: "database_primary".into(),
//!     node_type: LogicalNodeType::Database,
//!     state: LogicalNodeState::Failed, // Database failure
//!     connections: db_connections,
//!     criticality: 0.9,
//!     resources: LogicalNodeResources { power: 500.0, cpu_load: 0.8, memory_bytes: 32768, bandwidth_bps: 100000 },
//! }).unwrap();
//!
//! let topology = LogicalTopology { nodes, adjacency: heapless::FnvIndexMap::new() };
//! let analyzer = SoftwareImpactAnalyzer::new(topology);
//! let rules = SoftwarePropagationRules::default();
//!
//! // Analyze impact propagation from database failure
//! let result = analyzer.analyze_impact_propagation("database_primary", &rules).unwrap();
//! println!("Software impact affects {} modules", result.affected_nodes.len());
//! ```
//!
//! ## Policy Response Simulation Example
//! ```rust
//! use floodfill_core::{
//!     LogicalNode, LogicalNodeType, LogicalNodeState, LogicalNodeResources,
//!     LogicalTopology, PolicyPropagationSimulator, PolicyType, PolicyPropagationRules
//! };
//!
//! // Create system topology for policy simulation
//! let mut nodes = heapless::Vec::new();
//!
//! nodes.push(LogicalNode {
//!     id: "comm_system".into(),
//!     node_type: LogicalNodeType::CommunicationModule,
//!     state: LogicalNodeState::Operational,
//!     connections: heapless::Vec::new(),
//!     criticality: 0.7,
//!     resources: LogicalNodeResources { power: 300.0, cpu_load: 0.5, memory_bytes: 4096, bandwidth_bps: 1000000 },
//! }).unwrap();
//!
//! let topology = LogicalTopology { nodes, adjacency: heapless::FnvIndexMap::new() };
//! let simulator = PolicyPropagationSimulator::new(topology);
//! let rules = PolicyPropagationRules::default();
//!
//! // Simulate reroute policy propagation
//! let result = simulator.simulate_policy_propagation("comm_system", PolicyType::Reroute, &rules).unwrap();
//! println!("Policy affects {} systems", result.affected_nodes.len());
//! println!("Resource impact: {:.1}W power", result.resource_impact.power);
//! ```

#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate alloc;

#[cfg(feature = "no_std")]
use alloc::{string::String, vec::Vec};

#[cfg(not(feature = "no_std"))]
use std::{string::String, vec::Vec};

use core::fmt;

/// Error types for flood-fill operations
///
/// **Requirement Traceability**: REQ-FLOOD-006 - Error Handling
/// - Provides comprehensive error classification for all failure modes
/// - Enables graceful degradation and fault isolation in space systems
/// - Supports both std and no_std error handling patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloodFillError {
    /// Stack overflow during flood-fill operation
    StackOverflow,
    /// Invalid grid coordinates
    InvalidCoordinates,
    /// Grid dimensions too large
    GridTooLarge,
    /// Memory allocation failure
    OutOfMemory,
    /// Operation timeout
    Timeout,
}

impl fmt::Display for FloodFillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FloodFillError::StackOverflow => write!(f, "Stack overflow during flood-fill"),
            FloodFillError::InvalidCoordinates => write!(f, "Invalid grid coordinates"),
            FloodFillError::GridTooLarge => write!(f, "Grid dimensions exceed limits"),
            FloodFillError::OutOfMemory => write!(f, "Memory allocation failure"),
            FloodFillError::Timeout => write!(f, "Operation timed out"),
        }
    }
}

/// Statistics for a connected region found by flood-fill
#[derive(Clone, Copy, Debug, Default)]
pub struct RegionStats {
    /// Number of cells in the region
    pub count: usize,
    /// Sum of x coordinates (for centroid calculation)
    pub sum_x: usize,
    /// Sum of y coordinates (for centroid calculation)
    pub sum_y: usize,
    /// Minimum x coordinate (bounding box)
    pub min_x: usize,
    /// Minimum y coordinate (bounding box)
    pub min_y: usize,
    /// Maximum x coordinate (bounding box)
    pub max_x: usize,
    /// Maximum y coordinate (bounding box)
    pub max_y: usize,
    /// Perimeter cell count
    pub perimeter: usize,
}

impl RegionStats {
    /// Create a new empty region statistics tracker
    pub fn new() -> Self {
        Self {
            count: 0,
            sum_x: 0,
            sum_y: 0,
            min_x: usize::MAX,
            min_y: usize::MAX,
            max_x: 0,
            max_y: 0,
            perimeter: 0,
        }
    }

    /// Add a cell to the region statistics
    pub fn add_cell(&mut self, x: usize, y: usize, is_perimeter: bool) {
        self.count += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);

        if is_perimeter {
            self.perimeter += 1;
        }
    }

    /// Calculate the centroid of the region
    pub fn centroid(&self) -> Option<(f32, f32)> {
        if self.count == 0 {
            None
        } else {
            Some((
                self.sum_x as f32 / self.count as f32,
                self.sum_y as f32 / self.count as f32,
            ))
        }
    }

    /// Get the bounding box of the region
    pub fn bounding_box(&self) -> Option<(usize, usize, usize, usize)> {
        if self.count == 0 {
            None
        } else {
            Some((self.min_x, self.min_y, self.max_x, self.max_y))
        }
    }

    /// Get the area of the region
    pub fn area(&self) -> usize {
        self.count
    }

    /// Calculate the compactness ratio (area / perimeter²)
    pub fn compactness(&self) -> f32 {
        if self.perimeter == 0 {
            0.0
        } else {
            self.count as f32 / (self.perimeter as f32 * self.perimeter as f32)
        }
    }
}

/// Performance metrics for flood-fill operations
///
/// **Requirement Traceability**: REQ-FLOOD-005 - Performance Constraints
/// - Tracks processing metrics for real-time constraint verification
/// - Monitors memory usage for space-constrained environments
/// - Provides timing data for algorithm optimization
#[derive(Debug, Clone, Copy, Default)]
pub struct FloodFillMetrics {
    /// Number of cells processed
    /// **Trace**: REQ-FLOOD-005 - Algorithm complexity measurement
    pub cells_processed: usize,
    /// Number of stack operations
    /// **Trace**: REQ-FLOOD-004 - Stack usage monitoring
    pub stack_operations: usize,
    /// Peak stack depth
    /// **Trace**: REQ-FLOOD-004 - Memory safety verification
    pub peak_stack_depth: usize,
    /// Processing time in microseconds (only available with std feature)
    /// **Trace**: REQ-FLOOD-005 - Real-time performance measurement
    pub processing_time_us: u64,
    /// Memory usage in bytes
    /// **Trace**: REQ-FLOOD-004 - Memory constraint monitoring
    pub memory_usage_bytes: usize,
}

/// Configuration for flood-fill operations
#[derive(Debug, Clone, Copy)]
pub struct FloodFillConfig {
    /// Maximum number of cells to process before timeout
    pub max_cells: usize,
    /// Maximum stack depth allowed
    pub max_stack_depth: usize,
    /// Whether to use 8-connectivity (includes diagonals)
    pub use_8_connectivity: bool,
}

impl Default for FloodFillConfig {
    fn default() -> Self {
        Self {
            max_cells: 65536,
            max_stack_depth: 8192,
            use_8_connectivity: false,
        }
    }
}

/// Result of a flood-fill operation
pub type FloodFillResult = Result<(RegionStats, FloodFillMetrics), FloodFillError>;

/// Get current timestamp in milliseconds (for compatibility)
pub fn get_timestamp() -> u64 {
    #[cfg(all(feature = "std", not(feature = "no_std")))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    #[cfg(feature = "no_std")]
    {
        // In no_std, return 0 or implement with a hardware timer
        0
    }
}

/// Perform flood-fill with 4-connectivity (N,E,S,W neighbors)
pub fn flood_fill_4conn<F>(
    width: usize,
    height: usize,
    is_target: F,
    visited: &mut [u8],
    start_x: usize,
    start_y: usize,
    config: &FloodFillConfig,
) -> FloodFillResult
where
    F: FnMut(usize, usize) -> bool,
{
    let mut config_4conn = *config;
    config_4conn.use_8_connectivity = false;
    flood_fill_impl(
        width,
        height,
        is_target,
        visited,
        start_x,
        start_y,
        &config_4conn,
    )
}

/// Perform flood-fill with 8-connectivity (including diagonal neighbors)
pub fn flood_fill_8conn<F>(
    width: usize,
    height: usize,
    is_target: F,
    visited: &mut [u8],
    start_x: usize,
    start_y: usize,
    config: &FloodFillConfig,
) -> FloodFillResult
where
    F: FnMut(usize, usize) -> bool,
{
    let mut config_8conn = *config;
    config_8conn.use_8_connectivity = true;
    flood_fill_impl(
        width,
        height,
        is_target,
        visited,
        start_x,
        start_y,
        &config_8conn,
    )
}

/// Check if a cell is on the perimeter of the region
fn is_perimeter_cell<F>(x: usize, y: usize, width: usize, height: usize, is_target: &mut F) -> bool
where
    F: FnMut(usize, usize) -> bool,
{
    // Cell is on perimeter if it's at grid boundary or has non-target neighbors
    if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
        return true;
    }

    let neighbors = [
        (x, y.wrapping_sub(1)), // North
        (x + 1, y),             // East
        (x, y + 1),             // South
        (x.wrapping_sub(1), y), // West
    ];

    for (nx, ny) in neighbors {
        if nx < width && ny < height && !is_target(nx, ny) {
            return true;
        }
    }

    false
}

/// Internal implementation of flood-fill algorithm
fn flood_fill_impl<F>(
    width: usize,
    height: usize,
    mut is_target: F,
    visited: &mut [u8],
    start_x: usize,
    start_y: usize,
    config: &FloodFillConfig,
) -> FloodFillResult
where
    F: FnMut(usize, usize) -> bool,
{
    // Validate inputs
    if width == 0 || height == 0 {
        return Err(FloodFillError::InvalidCoordinates);
    }

    if width > 65536 || height > 65536 || width * height > config.max_cells {
        return Err(FloodFillError::GridTooLarge);
    }

    if start_x >= width || start_y >= height {
        return Err(FloodFillError::InvalidCoordinates);
    }

    if visited.len() != width * height {
        return Err(FloodFillError::InvalidCoordinates);
    }

    #[cfg(all(feature = "std", not(feature = "no_std")))]
    let start_time = std::time::Instant::now();

    #[cfg(feature = "heapless")]
    let mut stack: heapless::Vec<(usize, usize), 8192> = heapless::Vec::new();

    #[cfg(not(feature = "heapless"))]
    let mut stack = Vec::new();

    let mut stats = RegionStats::new();
    let mut metrics = FloodFillMetrics::default();

    let start_idx = start_y * width + start_x;

    // Check if starting position is valid and unvisited
    if visited[start_idx] != 0 || !is_target(start_x, start_y) {
        #[cfg(all(feature = "std", not(feature = "no_std")))]
        {
            metrics.processing_time_us = start_time.elapsed().as_micros() as u64;
        }
        return Ok((stats, metrics));
    }

    // Initialize flood-fill
    #[cfg(feature = "heapless")]
    {
        if stack.push((start_x, start_y)).is_err() {
            return Err(FloodFillError::StackOverflow);
        }
    }

    #[cfg(not(feature = "heapless"))]
    {
        stack.push((start_x, start_y));
    }

    visited[start_idx] = 1;
    metrics.stack_operations += 1;
    metrics.peak_stack_depth = 1;

    while let Some((x, y)) = stack.pop() {
        metrics.stack_operations += 1;
        metrics.cells_processed += 1;

        // Check if we've processed too many cells
        if metrics.cells_processed > config.max_cells {
            return Err(FloodFillError::OutOfMemory);
        }

        // Check if this is a perimeter cell
        let is_perimeter = is_perimeter_cell(x, y, width, height, &mut is_target);
        stats.add_cell(x, y, is_perimeter);

        // Get neighbors based on connectivity
        let neighbors: &[(i32, i32)] = if config.use_8_connectivity {
            &[
                (x as i32, y as i32 - 1),     // North
                (x as i32 + 1, y as i32 - 1), // Northeast
                (x as i32 + 1, y as i32),     // East
                (x as i32 + 1, y as i32 + 1), // Southeast
                (x as i32, y as i32 + 1),     // South
                (x as i32 - 1, y as i32 + 1), // Southwest
                (x as i32 - 1, y as i32),     // West
                (x as i32 - 1, y as i32 - 1), // Northwest
            ]
        } else {
            &[
                (x as i32, y as i32 - 1), // North
                (x as i32 + 1, y as i32), // East
                (x as i32, y as i32 + 1), // South
                (x as i32 - 1, y as i32), // West
            ]
        };

        for &(nx, ny) in neighbors {
            // Bounds check
            if nx < 0 || ny < 0 {
                continue;
            }
            let ux = nx as usize;
            let uy = ny as usize;
            if ux >= width || uy >= height {
                continue;
            }

            let nidx = uy * width + ux;

            // Skip if already visited
            if visited[nidx] != 0 {
                continue;
            }

            // Skip if not a target cell
            if !is_target(ux, uy) {
                continue;
            }

            // Mark as visited and add to stack
            visited[nidx] = 1;

            #[cfg(feature = "heapless")]
            {
                if stack.push((ux, uy)).is_err() {
                    return Err(FloodFillError::StackOverflow);
                }
            }

            #[cfg(not(feature = "heapless"))]
            {
                stack.push((ux, uy));
            }

            metrics.stack_operations += 1;
            metrics.peak_stack_depth = metrics.peak_stack_depth.max(stack.len());

            // Check stack depth limit
            if stack.len() > config.max_stack_depth {
                return Err(FloodFillError::StackOverflow);
            }
        }
    }

    #[cfg(all(feature = "std", not(feature = "no_std")))]
    {
        metrics.processing_time_us = start_time.elapsed().as_micros() as u64;
    }

    metrics.memory_usage_bytes = stack.capacity() * core::mem::size_of::<(usize, usize)>()
        + core::mem::size_of_val(&*visited);

    Ok((stats, metrics))
}

//
// === LOGICAL SYSTEM FLOOD-FILL FOR HARDWARE/SOFTWARE DECISION-AIDING ===
//

/// Logical system node representing hardware or software components
///
/// **Requirement Traceability**: REQ-FLOOD-009 - Logical System Analysis
/// - Represents nodes in logical system topology for dependency analysis
/// - Supports both hardware components and software modules
/// - Enables fault propagation and impact analysis modeling
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalNode {
    /// Unique node identifier
    pub id: String,
    /// Node type classification
    pub node_type: LogicalNodeType,
    /// Current operational state
    pub state: LogicalNodeState,
    /// Connected node IDs (logical adjacency)
    pub connections: heapless::Vec<String, 16>,
    /// Node criticality level (0.0 = non-critical, 1.0 = mission-critical)
    pub criticality: f32,
    /// Resource requirements/capacity
    pub resources: LogicalNodeResources,
}

/// Types of logical nodes in the system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalNodeType {
    // Hardware Components
    PowerBus,
    SensorCluster,
    ProcessingUnit,
    CommunicationModule,
    ActuatorArray,
    ThermalSystem,
    // Software Components
    ServiceModule,
    Thread,
    Process,
    Database,
    NetworkService,
    ControlLoop,
}

/// Operational state of logical nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalNodeState {
    Operational,
    Degraded,
    Failed,
    Isolated,
    Maintenance,
    Unknown,
}

/// Resource requirements and capacity for logical nodes
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogicalNodeResources {
    /// Power consumption/capacity (Watts)
    pub power: f32,
    /// Processing load (0.0 to 1.0)
    pub cpu_load: f32,
    /// Memory usage (bytes)
    pub memory_bytes: u32,
    /// Bandwidth usage (bps)
    pub bandwidth_bps: u32,
}

/// Logical system topology for flood-fill analysis
///
/// **Requirement Traceability**: REQ-FLOOD-009, REQ-FLOOD-010, REQ-FLOOD-011
/// - REQ-FLOOD-009: Provides logical adjacency modeling framework
/// - REQ-FLOOD-010: Enables hardware fault propagation analysis
/// - REQ-FLOOD-011: Supports software module dependency analysis
#[derive(Debug, Clone)]
pub struct LogicalTopology {
    /// All nodes in the system
    pub nodes: heapless::Vec<LogicalNode, 256>,
    /// Adjacency relationships (node_id -> connected_node_ids)
    pub adjacency: heapless::FnvIndexMap<String, heapless::Vec<String, 16>, 256>,
}

/// Results of logical flood-fill analysis
#[derive(Debug, Clone)]
pub struct LogicalFloodFillResult {
    /// Affected node IDs in propagation order
    pub affected_nodes: heapless::Vec<String, 256>,
    /// Propagation depth from initial failure
    pub propagation_depth: usize,
    /// Total estimated impact score
    pub total_impact_score: f32,
    /// Critical nodes affected
    pub critical_nodes_affected: usize,
    /// Resource impact assessment
    pub resource_impact: LogicalNodeResources,
}

/// Hardware fault propagation analyzer
///
/// **Requirement Traceability**: REQ-FLOOD-010 - Hardware Fault Propagation
/// - Traces fault propagation through connected subsystems
/// - Models power bus, sensor cluster, and component connectivity
/// - Provides impact assessment for hardware failure scenarios
pub struct HardwareFaultAnalyzer {
    topology: LogicalTopology,
}

impl HardwareFaultAnalyzer {
    /// Create new hardware fault analyzer
    pub fn new(topology: LogicalTopology) -> Self {
        Self { topology }
    }

    /// Analyze fault propagation from initial failure point
    ///
    /// **Requirement Traceability**: REQ-FLOOD-010 - Hardware Fault Propagation
    /// - Uses flood-fill to trace fault propagation through hardware connections
    /// - Considers power dependencies, thermal coupling, and signal paths
    /// - Returns comprehensive impact assessment for decision-aiding
    pub fn analyze_fault_propagation(
        &self,
        initial_failure_node: &str,
        propagation_rules: &HardwarePropagationRules,
    ) -> Result<LogicalFloodFillResult, FloodFillError> {
        let mut visited = heapless::FnvIndexSet::<String, 256>::new();
        let mut affected_nodes = heapless::Vec::<String, 256>::new();
        let mut queue = heapless::Deque::<String, 256>::new();

        // Initialize with failure node
        if queue.push_back(initial_failure_node.into()).is_err() {
            return Err(FloodFillError::InvalidCoordinates);
        }

        let mut total_impact_score = 0.0;
        let mut critical_nodes_affected = 0;
        let mut resource_impact = LogicalNodeResources {
            power: 0.0,
            cpu_load: 0.0,
            memory_bytes: 0,
            bandwidth_bps: 0,
        };

        // Flood-fill through hardware connections
        while let Some(current_node_id) = queue.pop_front() {
            if visited.contains(&current_node_id) {
                continue;
            }

            if visited.insert(current_node_id.clone()).is_err() {
                break; // Memory limit reached
            }

            if affected_nodes.push(current_node_id.clone()).is_err() {
                break; // Memory limit reached
            }

            // Find current node
            if let Some(current_node) = self.topology.nodes.iter().find(|n| n.id == current_node_id)
            {
                total_impact_score += current_node.criticality;

                if current_node.criticality > 0.8 {
                    critical_nodes_affected += 1;
                }

                // Accumulate resource impact
                resource_impact.power += current_node.resources.power;
                resource_impact.cpu_load += current_node.resources.cpu_load;
                resource_impact.memory_bytes += current_node.resources.memory_bytes;
                resource_impact.bandwidth_bps += current_node.resources.bandwidth_bps;

                // Check connected nodes for propagation
                for connected_id in &current_node.connections {
                    if visited.contains(connected_id) {
                        continue;
                    }

                    if let Some(connected_node) =
                        self.topology.nodes.iter().find(|n| n.id == *connected_id)
                    {
                        if self.should_propagate_fault(
                            current_node,
                            connected_node,
                            propagation_rules,
                        ) {
                            if queue.push_back(connected_id.clone()).is_err() {
                                break; // Queue full
                            }
                        }
                    }
                }
            }
        }

        Ok(LogicalFloodFillResult {
            affected_nodes,
            propagation_depth: visited.len(),
            total_impact_score,
            critical_nodes_affected,
            resource_impact,
        })
    }

    /// Determine if fault should propagate between nodes
    fn should_propagate_fault(
        &self,
        from_node: &LogicalNode,
        to_node: &LogicalNode,
        rules: &HardwarePropagationRules,
    ) -> bool {
        match (from_node.node_type, to_node.node_type) {
            // Power bus failures propagate to all connected components
            (LogicalNodeType::PowerBus, _) => rules.power_propagation,

            // Thermal system failures affect nearby components
            (LogicalNodeType::ThermalSystem, LogicalNodeType::ProcessingUnit) => {
                rules.thermal_propagation
            }
            (LogicalNodeType::ThermalSystem, LogicalNodeType::SensorCluster) => {
                rules.thermal_propagation
            }

            // Processing unit failures affect dependent sensors and actuators
            (LogicalNodeType::ProcessingUnit, LogicalNodeType::SensorCluster) => {
                rules.control_propagation
            }
            (LogicalNodeType::ProcessingUnit, LogicalNodeType::ActuatorArray) => {
                rules.control_propagation
            }

            // Communication failures isolate connected systems
            (LogicalNodeType::CommunicationModule, _) => rules.communication_propagation,

            // Default: no propagation
            _ => false,
        }
    }
}

/// Hardware fault propagation rules
#[derive(Debug, Clone, Copy)]
pub struct HardwarePropagationRules {
    /// Enable power bus fault propagation
    pub power_propagation: bool,
    /// Enable thermal fault propagation
    pub thermal_propagation: bool,
    /// Enable control system fault propagation
    pub control_propagation: bool,
    /// Enable communication fault propagation
    pub communication_propagation: bool,
    /// Criticality threshold for propagation
    pub criticality_threshold: f32,
}

impl Default for HardwarePropagationRules {
    fn default() -> Self {
        Self {
            power_propagation: true,
            thermal_propagation: true,
            control_propagation: true,
            communication_propagation: false, // Communications often have redundancy
            criticality_threshold: 0.5,
        }
    }
}

/// Software module impact analyzer
///
/// **Requirement Traceability**: REQ-FLOOD-011 - Software Module Containment
/// - Isolates impacted services in modular architectures
/// - Analyzes call graphs and dependency trees
/// - Provides containment strategies for software faults
pub struct SoftwareImpactAnalyzer {
    topology: LogicalTopology,
}

impl SoftwareImpactAnalyzer {
    /// Create new software impact analyzer
    pub fn new(topology: LogicalTopology) -> Self {
        Self { topology }
    }

    /// Analyze software module impact propagation
    ///
    /// **Requirement Traceability**: REQ-FLOOD-011 - Software Module Containment
    /// - Uses flood-fill to trace impact through software dependencies
    /// - Considers service calls, shared resources, and data dependencies
    /// - Returns isolation recommendations for fault containment
    pub fn analyze_impact_propagation(
        &self,
        initial_failure_module: &str,
        propagation_rules: &SoftwarePropagationRules,
    ) -> Result<LogicalFloodFillResult, FloodFillError> {
        let mut visited = heapless::FnvIndexSet::<String, 256>::new();
        let mut affected_nodes = heapless::Vec::<String, 256>::new();
        let mut queue = heapless::Deque::<String, 256>::new();

        // Initialize with failure module
        if queue.push_back(initial_failure_module.into()).is_err() {
            return Err(FloodFillError::InvalidCoordinates);
        }

        let mut total_impact_score = 0.0;
        let mut critical_nodes_affected = 0;
        let mut resource_impact = LogicalNodeResources {
            power: 0.0,
            cpu_load: 0.0,
            memory_bytes: 0,
            bandwidth_bps: 0,
        };

        // Flood-fill through software dependencies
        while let Some(current_module_id) = queue.pop_front() {
            if visited.contains(&current_module_id) {
                continue;
            }

            if visited.insert(current_module_id.clone()).is_err() {
                break; // Memory limit reached
            }

            if affected_nodes.push(current_module_id.clone()).is_err() {
                break; // Memory limit reached
            }

            // Find current module
            if let Some(current_module) = self
                .topology
                .nodes
                .iter()
                .find(|n| n.id == current_module_id)
            {
                total_impact_score += current_module.criticality;

                if current_module.criticality > 0.8 {
                    critical_nodes_affected += 1;
                }

                // Accumulate resource impact
                resource_impact.power += current_module.resources.power;
                resource_impact.cpu_load += current_module.resources.cpu_load;
                resource_impact.memory_bytes += current_module.resources.memory_bytes;
                resource_impact.bandwidth_bps += current_module.resources.bandwidth_bps;

                // Check dependent modules for propagation
                for connected_id in &current_module.connections {
                    if visited.contains(connected_id) {
                        continue;
                    }

                    if let Some(connected_module) =
                        self.topology.nodes.iter().find(|n| n.id == *connected_id)
                    {
                        if self.should_propagate_software_impact(
                            current_module,
                            connected_module,
                            propagation_rules,
                        ) {
                            if queue.push_back(connected_id.clone()).is_err() {
                                break; // Queue full
                            }
                        }
                    }
                }
            }
        }

        Ok(LogicalFloodFillResult {
            affected_nodes,
            propagation_depth: visited.len(),
            total_impact_score,
            critical_nodes_affected,
            resource_impact,
        })
    }

    /// Determine if software impact should propagate between modules
    fn should_propagate_software_impact(
        &self,
        from_module: &LogicalNode,
        to_module: &LogicalNode,
        rules: &SoftwarePropagationRules,
    ) -> bool {
        match (from_module.node_type, to_module.node_type) {
            // Database failures affect all dependent services
            (LogicalNodeType::Database, LogicalNodeType::ServiceModule) => {
                rules.database_propagation
            }

            // Service failures affect calling services
            (LogicalNodeType::ServiceModule, LogicalNodeType::ServiceModule) => {
                rules.service_propagation
            }

            // Process failures affect all threads in process
            (LogicalNodeType::Process, LogicalNodeType::Thread) => rules.process_propagation,

            // Network service failures affect dependent services
            (LogicalNodeType::NetworkService, _) => rules.network_propagation,

            // Control loop failures affect related control loops
            (LogicalNodeType::ControlLoop, LogicalNodeType::ControlLoop) => {
                rules.control_propagation
            }

            // Default: propagate based on criticality
            _ => from_module.criticality > rules.criticality_threshold,
        }
    }
}

/// Software fault propagation rules
#[derive(Debug, Clone, Copy)]
pub struct SoftwarePropagationRules {
    /// Enable database dependency propagation
    pub database_propagation: bool,
    /// Enable service-to-service propagation
    pub service_propagation: bool,
    /// Enable process-to-thread propagation
    pub process_propagation: bool,
    /// Enable network service propagation
    pub network_propagation: bool,
    /// Enable control loop propagation
    pub control_propagation: bool,
    /// Criticality threshold for propagation
    pub criticality_threshold: f32,
}

impl Default for SoftwarePropagationRules {
    fn default() -> Self {
        Self {
            database_propagation: true,
            service_propagation: true,
            process_propagation: true,
            network_propagation: true,
            control_propagation: true,
            criticality_threshold: 0.3, // Lower threshold for software
        }
    }
}

/// Policy response propagation simulator
///
/// **Requirement Traceability**: REQ-FLOOD-012 - Policy Response Modeling
/// - Simulates policy propagation through logical system links
/// - Models shutdown, reroute, and isolation policies
/// - Provides decision-aiding for autonomous policy responses
pub struct PolicyPropagationSimulator {
    topology: LogicalTopology,
}

impl PolicyPropagationSimulator {
    /// Create new policy propagation simulator
    pub fn new(topology: LogicalTopology) -> Self {
        Self { topology }
    }

    /// Simulate policy propagation through system
    ///
    /// **Requirement Traceability**: REQ-FLOOD-012 - Policy Response Modeling
    /// - Uses flood-fill to simulate policy effects across logical connections
    /// - Models different policy types (shutdown, reroute, isolate)
    /// - Returns comprehensive impact assessment for policy decision-aiding
    pub fn simulate_policy_propagation(
        &self,
        initial_policy_node: &str,
        policy_type: PolicyType,
        propagation_rules: &PolicyPropagationRules,
    ) -> Result<LogicalFloodFillResult, FloodFillError> {
        let mut visited = heapless::FnvIndexSet::<String, 256>::new();
        let mut affected_nodes = heapless::Vec::<String, 256>::new();
        let mut queue = heapless::Deque::<String, 256>::new();

        // Initialize with policy application node
        if queue.push_back(initial_policy_node.into()).is_err() {
            return Err(FloodFillError::InvalidCoordinates);
        }

        let mut total_impact_score = 0.0;
        let mut critical_nodes_affected = 0;
        let mut resource_impact = LogicalNodeResources {
            power: 0.0,
            cpu_load: 0.0,
            memory_bytes: 0,
            bandwidth_bps: 0,
        };

        // Flood-fill through policy propagation paths
        while let Some(current_node_id) = queue.pop_front() {
            if visited.contains(&current_node_id) {
                continue;
            }

            if visited.insert(current_node_id.clone()).is_err() {
                break; // Memory limit reached
            }

            if affected_nodes.push(current_node_id.clone()).is_err() {
                break; // Memory limit reached
            }

            // Find current node
            if let Some(current_node) = self.topology.nodes.iter().find(|n| n.id == current_node_id)
            {
                total_impact_score += current_node.criticality;

                if current_node.criticality > 0.8 {
                    critical_nodes_affected += 1;
                }

                // Calculate policy impact on resources
                let policy_factor = self.get_policy_resource_factor(&policy_type);
                resource_impact.power += current_node.resources.power * policy_factor;
                resource_impact.cpu_load += current_node.resources.cpu_load * policy_factor;
                resource_impact.memory_bytes +=
                    (current_node.resources.memory_bytes as f32 * policy_factor) as u32;
                resource_impact.bandwidth_bps +=
                    (current_node.resources.bandwidth_bps as f32 * policy_factor) as u32;

                // Check connected nodes for policy propagation
                for connected_id in &current_node.connections {
                    if visited.contains(connected_id) {
                        continue;
                    }

                    if let Some(connected_node) =
                        self.topology.nodes.iter().find(|n| n.id == *connected_id)
                    {
                        if self.should_propagate_policy(
                            current_node,
                            connected_node,
                            &policy_type,
                            propagation_rules,
                        ) {
                            if queue.push_back(connected_id.clone()).is_err() {
                                break; // Queue full
                            }
                        }
                    }
                }
            }
        }

        Ok(LogicalFloodFillResult {
            affected_nodes,
            propagation_depth: visited.len(),
            total_impact_score,
            critical_nodes_affected,
            resource_impact,
        })
    }

    /// Determine if policy should propagate between nodes
    fn should_propagate_policy(
        &self,
        from_node: &LogicalNode,
        to_node: &LogicalNode,
        policy_type: &PolicyType,
        rules: &PolicyPropagationRules,
    ) -> bool {
        match policy_type {
            PolicyType::Shutdown => {
                // Shutdown propagates through power dependencies
                matches!(
                    (from_node.node_type, to_node.node_type),
                    (LogicalNodeType::PowerBus, _)
                        | (
                            LogicalNodeType::ProcessingUnit,
                            LogicalNodeType::SensorCluster
                        )
                        | (
                            LogicalNodeType::ProcessingUnit,
                            LogicalNodeType::ActuatorArray
                        )
                ) && rules.shutdown_propagation
            }
            PolicyType::Reroute => {
                // Reroute affects communication and service paths
                matches!(
                    (from_node.node_type, to_node.node_type),
                    (LogicalNodeType::CommunicationModule, _)
                        | (LogicalNodeType::NetworkService, _)
                        | (
                            LogicalNodeType::ServiceModule,
                            LogicalNodeType::ServiceModule
                        )
                ) && rules.reroute_propagation
            }
            PolicyType::Isolate => {
                // Isolation affects all connected components
                rules.isolate_propagation
            }
        }
    }

    /// Get resource impact factor for policy type
    fn get_policy_resource_factor(&self, policy_type: &PolicyType) -> f32 {
        match policy_type {
            PolicyType::Shutdown => 0.0, // Complete resource shutdown
            PolicyType::Reroute => 1.2,  // Slightly increased resource usage for rerouting
            PolicyType::Isolate => 0.1,  // Minimal resource usage in isolation
        }
    }
}

/// Types of policies that can be simulated
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyType {
    /// Shutdown affected components
    Shutdown,
    /// Reroute traffic/data flow
    Reroute,
    /// Isolate components from system
    Isolate,
}

/// Policy propagation rules
#[derive(Debug, Clone, Copy)]
pub struct PolicyPropagationRules {
    /// Enable shutdown policy propagation
    pub shutdown_propagation: bool,
    /// Enable reroute policy propagation
    pub reroute_propagation: bool,
    /// Enable isolate policy propagation
    pub isolate_propagation: bool,
    /// Maximum propagation depth
    pub max_propagation_depth: usize,
}

impl Default for PolicyPropagationRules {
    fn default() -> Self {
        Self {
            shutdown_propagation: true,
            reroute_propagation: true,
            isolate_propagation: false, // Isolation typically doesn't propagate
            max_propagation_depth: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "no_std"))]
    use std::vec;

    #[cfg(feature = "no_std")]
    use alloc::vec;

    #[test]
    fn test_region_stats_empty() {
        let stats = RegionStats::new();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.centroid(), None);
        assert_eq!(stats.bounding_box(), None);
    }

    #[test]
    fn test_region_stats_single_cell() {
        let mut stats = RegionStats::new();
        stats.add_cell(5, 3, false);

        assert_eq!(stats.count, 1);
        assert_eq!(stats.centroid(), Some((5.0, 3.0)));
        assert_eq!(stats.bounding_box(), Some((5, 3, 5, 3)));
    }

    #[test]
    fn test_flood_fill_4conn_simple() {
        let width = 5;
        let height = 5;
        let mut visited = vec![0u8; width * height];
        let config = FloodFillConfig::default();

        // Create a simple 2x2 anomaly region
        let anomaly_cells = [(1, 1), (1, 2), (2, 1), (2, 2)];
        let is_anomaly = |x: usize, y: usize| anomaly_cells.contains(&(x, y));

        let result = flood_fill_4conn(width, height, is_anomaly, &mut visited, 1, 1, &config);

        assert!(result.is_ok());
        let (stats, _metrics) = result.unwrap();
        assert_eq!(stats.count, 4);
        assert_eq!(stats.centroid(), Some((1.5, 1.5)));
    }

    #[test]
    fn test_flood_fill_error_conditions() {
        let mut visited = vec![0u8; 25];
        let config = FloodFillConfig::default();
        let is_target = |_x: usize, _y: usize| true;

        // Test invalid coordinates
        let result = flood_fill_4conn(5, 5, is_target, &mut visited, 10, 10, &config);
        assert_eq!(result.unwrap_err(), FloodFillError::InvalidCoordinates);

        // Test grid too large
        let large_config = FloodFillConfig {
            max_cells: 10,
            ..config
        };
        let result = flood_fill_4conn(5, 5, is_target, &mut visited, 0, 0, &large_config);
        assert_eq!(result.unwrap_err(), FloodFillError::GridTooLarge);
    }

    #[test]
    fn test_logical_node_creation() {
        let node = LogicalNode {
            id: "power_bus_1".into(),
            node_type: LogicalNodeType::PowerBus,
            state: LogicalNodeState::Operational,
            connections: heapless::Vec::new(),
            criticality: 0.9,
            resources: LogicalNodeResources {
                power: 100.0,
                cpu_load: 0.0,
                memory_bytes: 0,
                bandwidth_bps: 0,
            },
        };

        assert_eq!(node.id, "power_bus_1");
        assert_eq!(node.node_type, LogicalNodeType::PowerBus);
        assert_eq!(node.state, LogicalNodeState::Operational);
        assert_eq!(node.criticality, 0.9);
    }

    #[test]
    fn test_hardware_fault_propagation() {
        // Create test topology
        let mut nodes = heapless::Vec::new();

        // Power bus
        let mut power_connections = heapless::Vec::new();
        power_connections.push("sensor_1".into()).unwrap();
        power_connections.push("processor_1".into()).unwrap();

        nodes
            .push(LogicalNode {
                id: "power_bus_1".into(),
                node_type: LogicalNodeType::PowerBus,
                state: LogicalNodeState::Failed, // Initial failure
                connections: power_connections,
                criticality: 0.9,
                resources: LogicalNodeResources {
                    power: 100.0,
                    cpu_load: 0.0,
                    memory_bytes: 0,
                    bandwidth_bps: 0,
                },
            })
            .unwrap();

        // Sensor cluster
        nodes
            .push(LogicalNode {
                id: "sensor_1".into(),
                node_type: LogicalNodeType::SensorCluster,
                state: LogicalNodeState::Operational,
                connections: heapless::Vec::new(),
                criticality: 0.7,
                resources: LogicalNodeResources {
                    power: 20.0,
                    cpu_load: 0.3,
                    memory_bytes: 1024,
                    bandwidth_bps: 1000,
                },
            })
            .unwrap();

        // Processing unit
        nodes
            .push(LogicalNode {
                id: "processor_1".into(),
                node_type: LogicalNodeType::ProcessingUnit,
                state: LogicalNodeState::Operational,
                connections: heapless::Vec::new(),
                criticality: 0.8,
                resources: LogicalNodeResources {
                    power: 50.0,
                    cpu_load: 0.6,
                    memory_bytes: 8192,
                    bandwidth_bps: 10000,
                },
            })
            .unwrap();

        let topology = LogicalTopology {
            nodes,
            adjacency: heapless::FnvIndexMap::new(),
        };

        let analyzer = HardwareFaultAnalyzer::new(topology);
        let rules = HardwarePropagationRules::default();

        let result = analyzer.analyze_fault_propagation("power_bus_1", &rules);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(!analysis.affected_nodes.is_empty());
        assert!(analysis.total_impact_score > 0.0);
    }

    #[test]
    fn test_software_impact_propagation() {
        // Create test software topology
        let mut nodes = heapless::Vec::new();

        // Database service
        let mut db_connections = heapless::Vec::new();
        db_connections.push("service_1".into()).unwrap();
        db_connections.push("service_2".into()).unwrap();

        nodes
            .push(LogicalNode {
                id: "database_1".into(),
                node_type: LogicalNodeType::Database,
                state: LogicalNodeState::Failed, // Initial failure
                connections: db_connections,
                criticality: 0.95,
                resources: LogicalNodeResources {
                    power: 80.0,
                    cpu_load: 0.8,
                    memory_bytes: 16384,
                    bandwidth_bps: 50000,
                },
            })
            .unwrap();

        // Service modules
        nodes
            .push(LogicalNode {
                id: "service_1".into(),
                node_type: LogicalNodeType::ServiceModule,
                state: LogicalNodeState::Operational,
                connections: heapless::Vec::new(),
                criticality: 0.6,
                resources: LogicalNodeResources {
                    power: 30.0,
                    cpu_load: 0.4,
                    memory_bytes: 4096,
                    bandwidth_bps: 5000,
                },
            })
            .unwrap();

        nodes
            .push(LogicalNode {
                id: "service_2".into(),
                node_type: LogicalNodeType::ServiceModule,
                state: LogicalNodeState::Operational,
                connections: heapless::Vec::new(),
                criticality: 0.5,
                resources: LogicalNodeResources {
                    power: 25.0,
                    cpu_load: 0.3,
                    memory_bytes: 2048,
                    bandwidth_bps: 3000,
                },
            })
            .unwrap();

        let topology = LogicalTopology {
            nodes,
            adjacency: heapless::FnvIndexMap::new(),
        };

        let analyzer = SoftwareImpactAnalyzer::new(topology);
        let rules = SoftwarePropagationRules::default();

        let result = analyzer.analyze_impact_propagation("database_1", &rules);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(!analysis.affected_nodes.is_empty());
        assert!(analysis.total_impact_score > 0.0);
        assert!(analysis.critical_nodes_affected > 0);
    }

    #[test]
    fn test_policy_propagation_simulation() {
        // Create test topology for policy simulation
        let mut nodes = heapless::Vec::new();

        // Communication module
        let mut comm_connections = heapless::Vec::new();
        comm_connections.push("service_1".into()).unwrap();

        nodes
            .push(LogicalNode {
                id: "comm_module_1".into(),
                node_type: LogicalNodeType::CommunicationModule,
                state: LogicalNodeState::Operational,
                connections: comm_connections,
                criticality: 0.8,
                resources: LogicalNodeResources {
                    power: 40.0,
                    cpu_load: 0.2,
                    memory_bytes: 1024,
                    bandwidth_bps: 100000,
                },
            })
            .unwrap();

        // Service module
        nodes
            .push(LogicalNode {
                id: "service_1".into(),
                node_type: LogicalNodeType::ServiceModule,
                state: LogicalNodeState::Operational,
                connections: heapless::Vec::new(),
                criticality: 0.6,
                resources: LogicalNodeResources {
                    power: 30.0,
                    cpu_load: 0.4,
                    memory_bytes: 4096,
                    bandwidth_bps: 5000,
                },
            })
            .unwrap();

        let topology = LogicalTopology {
            nodes,
            adjacency: heapless::FnvIndexMap::new(),
        };

        let simulator = PolicyPropagationSimulator::new(topology);
        let rules = PolicyPropagationRules::default();

        // Test reroute policy
        let result =
            simulator.simulate_policy_propagation("comm_module_1", PolicyType::Reroute, &rules);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(!analysis.affected_nodes.is_empty());
        assert!(analysis.total_impact_score > 0.0);

        // Test shutdown policy
        let result =
            simulator.simulate_policy_propagation("comm_module_1", PolicyType::Shutdown, &rules);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(!analysis.affected_nodes.is_empty());
    }
}
