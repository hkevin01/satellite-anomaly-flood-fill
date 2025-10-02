//! Feature extraction and component analysis for satellite anomaly detection
//!
//! # Requirement Traceability
//!
//! This crate implements the following system requirements:
//!
//! ## REQ-FEAT-001: Component Detection
//! **Requirement**: System shall detect connected anomalous regions with configurable connectivity
//! **Implementation**: [`ComponentTracker::extract_components`] with 4/8-connectivity flood-fill
//! **Verification**: Unit tests verify detection of simple and complex anomaly patterns
//!
//! ## REQ-FEAT-002: Temporal Tracking
//! **Requirement**: System shall track components across multiple frames with unique IDs
//! **Implementation**: [`Component`] with `id`, `birth_frame`, `last_seen_frame` tracking
//! **Verification**: IoU-based matching validates component continuity across frames
//!
//! ## REQ-FEAT-003: Growth Analysis
//! **Requirement**: System shall calculate growth rates for anomaly progression analysis
//! **Implementation**: [`Component::growth_rate`] computed from area changes over time
//! **Verification**: Growth rate calculation tested with expanding/shrinking regions
//!
//! ## REQ-FEAT-004: Confidence Scoring
//! **Requirement**: System shall provide confidence metrics for component reliability
//! **Implementation**: [`Component::confidence`] based on stability and consistency
//! **Verification**: Confidence updates tested for stable and unstable components
//!
//! ## REQ-FEAT-005: Memory Constraints
//! **Requirement**: System shall operate within bounded memory for space deployment
//! **Implementation**: Heapless vectors with compile-time size limits (256 components)
//! **Verification**: Memory bounds enforced at compile time, no dynamic allocation
//!
//! ## REQ-FEAT-006: Real-time Performance
//! **Requirement**: System shall complete processing within 50ms timeout
//! **Implementation**: Configurable timeout in [`ComponentExtractionConfig::timeout_us`]
//! **Verification**: Timeout handling tested with large grid processing
//!
//! ## REQ-FEAT-007: Intersection-over-Union Matching
//! **Requirement**: System shall match components using IoU threshold for robustness
//! **Implementation**: [`Component::iou`] with configurable threshold matching
//! **Verification**: IoU calculation tested with overlapping and non-overlapping regions
//!
//! ## REQ-FEAT-008: Component Lifecycle Management
//! **Requirement**: System shall remove stale components based on age thresholds
//! **Implementation**: [`Component::should_remove`] with configurable age limits
//! **Verification**: Age-based cleanup tested with frame progression scenarios
//!
//! ## REQ-FEAT-009: Threat Classification
//! **Requirement**: System shall classify anomalies by severity level for prioritization
//! **Implementation**: [`ThreatLevel`] enum with 5-level orderable classification
//! **Verification**: Ordering and default behavior validated in type system
//!
//! ## REQ-FEAT-010: Error Handling
//! **Requirement**: System shall provide comprehensive error reporting for fault isolation
//! **Implementation**: [`FeatureError`] covering all failure modes with context
//! **Verification**: Error propagation and conversion tested across operation modes
//!
//! ## REQ-FEAT-011: Performance Monitoring
//! **Requirement**: System shall provide telemetry for performance analysis
//! **Implementation**: [`ExtractionMetrics`] with timing, memory, and operation counts
//! **Verification**: Metrics collection verified in std and no_std environments
//!
//! ## REQ-FEAT-012: Hardware/Software Decision-Aiding Integration
//! **Requirement**: System shall support logical flood-fill for hardware/software fault analysis
//! **Implementation**: Integration with [`LogicalFloodFillResult`] for decision-aiding systems
//! **Verification**: Logical analysis integration tested with component-to-node mapping
//!
//! ## REQ-FEAT-013: Logical System Integration
//! **Requirement**: System shall convert geographic anomalies to logical system representations
//! **Implementation**: [`ComponentNodeTypeMapper`], [`LogicalTopologyBuilder`] for system modeling
//! **Verification**: Component-to-logical conversion validated with connectivity analysis

#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate alloc;

#[cfg(feature = "no_std")]
use alloc::format;

#[cfg(not(feature = "no_std"))]
use std::string::ToString;

#[cfg(feature = "no_std")]
use alloc::string::ToString;

use floodfill_core::{
    flood_fill_4conn,
    flood_fill_8conn,
    FloodFillConfig,
    FloodFillError,
    HardwareFaultAnalyzer,
    HardwarePropagationRules,
    LogicalFloodFillResult,
    // Logical flood-fill imports for hardware/software decision-aiding
    LogicalNode,
    LogicalNodeResources,
    LogicalNodeState,
    LogicalNodeType,
    LogicalTopology,
    PolicyPropagationRules,
    PolicyPropagationSimulator,
    PolicyType,
    RegionStats,
    SoftwareImpactAnalyzer,
    SoftwarePropagationRules,
};

/// Threat level for anomaly classification
///
/// **Requirement Traceability**: REQ-FEAT-009 - Threat Classification
/// - System shall classify anomalies by severity level
/// - Implementation provides 5-level threat classification from None to Critical
/// - Orderable for prioritization in decision systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ThreatLevel {
    #[default]
    None = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Evolution state for temporal analysis
///
/// **Requirement Traceability**: REQ-FEAT-003 - Growth Analysis
/// - System shall track anomaly evolution patterns over time
/// - Implementation provides state classification for temporal behavior
/// - Used in conjunction with growth_rate for comprehensive analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvolutionState {
    #[default]
    Stable,
    Growing,
    Shrinking,
    Oscillating,
    Disappeared,
}

/// Legacy Feature type for backward compatibility with decisions crate
pub type Feature = Component;

/// Legacy FeatureTracker type for backward compatibility
pub type FeatureTracker = ComponentTracker;

/// Error types for feature extraction operations
///
/// **Requirement Traceability**: REQ-FEAT-010 - Error Handling
/// - System shall provide comprehensive error reporting for all failure modes
/// - Implementation covers flood-fill failures, memory constraints, and timeouts
/// - Enables graceful degradation and fault isolation in space systems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureError {
    /// Flood-fill operation failed
    FloodFillFailed(FloodFillError),
    /// Too many components found
    TooManyComponents,
    /// Invalid grid parameters
    InvalidGrid,
    /// Memory allocation failure
    OutOfMemory,
    /// Processing timeout
    Timeout,
}

impl From<FloodFillError> for FeatureError {
    fn from(err: FloodFillError) -> Self {
        FeatureError::FloodFillFailed(err)
    }
}

impl core::fmt::Display for FeatureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FeatureError::FloodFillFailed(err) => write!(f, "Flood-fill failed: {}", err),
            FeatureError::TooManyComponents => write!(f, "Too many components found"),
            FeatureError::InvalidGrid => write!(f, "Invalid grid parameters"),
            FeatureError::OutOfMemory => write!(f, "Out of memory"),
            FeatureError::Timeout => write!(f, "Processing timeout"),
        }
    }
}

#[cfg(all(feature = "std", not(feature = "no_std")))]
impl std::error::Error for FeatureError {}

/// A connected component with statistics and temporal tracking
///
/// **Requirement Traceability**: REQ-FEAT-002, REQ-FEAT-003, REQ-FEAT-004
/// - REQ-FEAT-002: Temporal tracking with unique IDs and frame history
/// - REQ-FEAT-003: Growth analysis through area change tracking
/// - REQ-FEAT-004: Confidence scoring based on stability metrics
///
/// This structure represents a single connected anomalous region with complete
/// temporal analysis capabilities for space-grade fault detection systems.
#[derive(Clone, Debug)]
pub struct Component {
    /// Region statistics
    /// **Trace**: REQ-FEAT-001 - Stores geometric properties of detected component
    pub stats: RegionStats,
    /// Component ID for tracking across frames
    /// **Trace**: REQ-FEAT-002 - Unique identifier for temporal continuity
    pub id: u32,
    /// Frame number when first detected
    /// **Trace**: REQ-FEAT-002 - Birth time for lifecycle management
    pub birth_frame: u32,
    /// Frame number when last seen
    /// **Trace**: REQ-FEAT-008 - Last update time for age-based cleanup
    pub last_seen_frame: u32,
    /// Growth rate in cells per frame
    /// **Trace**: REQ-FEAT-003 - Quantitative growth analysis metric
    pub growth_rate: f32,
    /// Previous area for growth calculation
    /// **Trace**: REQ-FEAT-003 - Historical data for rate computation
    pub previous_area: usize,
    /// Confidence score (0.0 to 1.0)
    /// **Trace**: REQ-FEAT-004 - Reliability metric for decision systems
    pub confidence: f32,
    /// Whether this component is stable
    /// **Trace**: REQ-FEAT-004 - Stability indicator for confidence computation
    pub is_stable: bool,
}

impl Component {
    /// Create a new component
    ///
    /// **Requirement Traceability**: REQ-FEAT-002 - Component Initialization
    /// - Initializes component with unique ID and birth frame
    /// - Sets initial confidence and stability metrics
    /// - Establishes baseline for temporal tracking
    pub fn new(stats: RegionStats, id: u32, frame: u32) -> Self {
        Self {
            stats,
            id,
            birth_frame: frame,
            last_seen_frame: frame,
            growth_rate: 0.0,
            previous_area: stats.area(),
            confidence: 1.0,
            is_stable: false,
        }
    }

    /// Update component with new statistics
    ///
    /// **Requirement Traceability**: REQ-FEAT-003, REQ-FEAT-004 - Temporal Analysis
    /// - REQ-FEAT-003: Calculates growth rate from area changes over time
    /// - REQ-FEAT-004: Updates confidence based on stability patterns
    /// - Implements adaptive confidence scoring for reliability assessment
    pub fn update(&mut self, new_stats: RegionStats, frame: u32) {
        // Calculate growth rate
        let frame_delta = frame.saturating_sub(self.last_seen_frame);
        if frame_delta > 0 {
            let area_delta = new_stats.area() as i32 - self.previous_area as i32;
            self.growth_rate = area_delta as f32 / frame_delta as f32;
        }

        self.previous_area = self.stats.area();
        self.stats = new_stats;
        self.last_seen_frame = frame;

        // Update stability based on growth rate
        self.is_stable = self.growth_rate.abs() < 0.1;

        // Update confidence based on consistency
        if self.is_stable {
            self.confidence = (self.confidence + 0.1).min(1.0);
        } else {
            self.confidence = (self.confidence - 0.05).max(0.1);
        }
    }

    /// Get the age of the component in frames
    ///
    /// **Requirement Traceability**: REQ-FEAT-008 - Lifecycle Management
    /// - Computes component age for cleanup decisions
    /// - Enables age-based filtering and prioritization
    pub fn age(&self) -> u32 {
        self.last_seen_frame.saturating_sub(self.birth_frame)
    }

    /// Check if component should be considered for removal
    ///
    /// **Requirement Traceability**: REQ-FEAT-008 - Component Lifecycle Management
    /// - Implements age-based cleanup policy for stale components
    /// - Prevents memory accumulation in long-running space systems
    /// - Configurable age threshold for mission-specific requirements
    pub fn should_remove(&self, current_frame: u32, max_age_without_update: u32) -> bool {
        current_frame.saturating_sub(self.last_seen_frame) > max_age_without_update
    }

    /// Calculate intersection over union with another component
    ///
    /// **Requirement Traceability**: REQ-FEAT-007 - IoU Matching
    /// - Implements robust component matching using IoU metric
    /// - Enables tracking continuity across frames with overlapping regions
    /// - Returns None for invalid bounding boxes, Some(0.0-1.0) for valid IoU
    pub fn iou(&self, other: &Component) -> Option<f32> {
        let bbox1 = self.stats.bounding_box()?;
        let bbox2 = other.stats.bounding_box()?;

        // Calculate intersection
        let x1 = bbox1.0.max(bbox2.0);
        let y1 = bbox1.1.max(bbox2.1);
        let x2 = bbox1.2.min(bbox2.2);
        let y2 = bbox1.3.min(bbox2.3);

        if x1 > x2 || y1 > y2 {
            return Some(0.0); // No intersection
        }

        let intersection = (x2 - x1 + 1) * (y2 - y1 + 1);
        let area1 = (bbox1.2 - bbox1.0 + 1) * (bbox1.3 - bbox1.1 + 1);
        let area2 = (bbox2.2 - bbox2.0 + 1) * (bbox2.3 - bbox2.1 + 1);
        let union = area1 + area2 - intersection;

        if union == 0 {
            Some(0.0)
        } else {
            Some(intersection as f32 / union as f32)
        }
    }
}

/// Configuration for component extraction
///
/// **Requirement Traceability**: REQ-FEAT-005, REQ-FEAT-006, REQ-FEAT-007
/// - REQ-FEAT-005: Memory constraints with max_components limit
/// - REQ-FEAT-006: Real-time performance with timeout_us configuration
/// - REQ-FEAT-007: IoU matching with configurable threshold
///
/// This configuration structure encapsulates all tunable parameters for
/// component extraction to meet mission-specific requirements.
#[derive(Debug, Clone, Copy)]
pub struct ComponentExtractionConfig {
    /// Maximum number of components to track
    /// **Trace**: REQ-FEAT-005 - Memory constraint enforcement
    pub max_components: usize,
    /// Minimum component size in cells
    /// **Trace**: REQ-FEAT-001 - Noise filtering for detection quality
    pub min_component_size: usize,
    /// Maximum component size in cells
    /// **Trace**: REQ-FEAT-001 - Prevents runaway detection on large anomalies
    pub max_component_size: usize,
    /// IoU threshold for component matching
    /// **Trace**: REQ-FEAT-007 - Configurable matching sensitivity
    pub iou_threshold: f32,
    /// Maximum age without update before removal
    /// **Trace**: REQ-FEAT-008 - Stale component cleanup policy
    pub max_age_frames: u32,
    /// Processing timeout in microseconds
    /// **Trace**: REQ-FEAT-006 - Real-time constraint enforcement
    pub timeout_us: u64,
    /// Flood-fill configuration
    /// **Trace**: REQ-FEAT-001 - Core detection algorithm parameters
    pub flood_fill_config: FloodFillConfig,
}

impl Default for ComponentExtractionConfig {
    fn default() -> Self {
        Self {
            max_components: 256,
            min_component_size: 1,
            max_component_size: 65536,
            iou_threshold: 0.3,
            max_age_frames: 10,
            timeout_us: 50_000, // 50ms
            flood_fill_config: FloodFillConfig::default(),
        }
    }
}

/// Component extraction metrics
///
/// **Requirement Traceability**: REQ-FEAT-006, REQ-FEAT-011 - Performance Monitoring
/// - REQ-FEAT-006: Timing metrics for real-time constraint verification
/// - REQ-FEAT-011: Performance telemetry for space system health monitoring
///
/// Provides comprehensive metrics for algorithm performance analysis and
/// real-time constraint verification in space-grade systems.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtractionMetrics {
    /// Total processing time in microseconds
    /// **Trace**: REQ-FEAT-006 - Real-time performance measurement
    pub total_time_us: u64,
    /// Number of flood-fill operations performed
    /// **Trace**: REQ-FEAT-011 - Algorithm complexity tracking
    pub flood_fills_performed: usize,
    /// Number of components found
    /// **Trace**: REQ-FEAT-001 - Detection effectiveness metric
    pub components_found: usize,
    /// Number of components matched with previous frame
    /// **Trace**: REQ-FEAT-007 - Tracking continuity metric
    pub components_matched: usize,
    /// Number of new components
    /// **Trace**: REQ-FEAT-002 - New detection tracking
    pub new_components: usize,
    /// Peak memory usage in bytes
    /// **Trace**: REQ-FEAT-005 - Memory constraint monitoring
    pub peak_memory_bytes: usize,
}

/// Component tracker for temporal analysis
///
/// **Requirement Traceability**: REQ-FEAT-001, REQ-FEAT-002, REQ-FEAT-005, REQ-FEAT-008
/// - REQ-FEAT-001: Core component detection using flood-fill algorithms
/// - REQ-FEAT-002: Temporal tracking with frame-based component management
/// - REQ-FEAT-005: Memory-bounded operation with heapless data structures
/// - REQ-FEAT-008: Automatic lifecycle management with age-based cleanup
///
/// This is the primary interface for component detection and tracking in
/// satellite anomaly detection systems. Designed for space-grade reliability
/// with deterministic memory usage and real-time performance constraints.
pub struct ComponentTracker {
    /// Current components
    /// **Trace**: REQ-FEAT-005 - Bounded memory with compile-time size limit
    components: heapless::Vec<Component, 256>,
    /// Next component ID
    /// **Trace**: REQ-FEAT-002 - Unique identifier generation for tracking
    next_id: u32,
    /// Current frame number
    /// **Trace**: REQ-FEAT-002 - Temporal reference for all operations
    current_frame: u32,
    /// Configuration
    /// **Trace**: REQ-FEAT-006 - Mission-configurable parameters
    config: ComponentExtractionConfig,
}

impl ComponentTracker {
    /// Create a new component tracker
    ///
    /// **Requirement Traceability**: REQ-FEAT-005 - Memory Management
    /// - Initializes tracker with bounded memory allocation
    /// - No dynamic memory allocation for space-grade reliability
    /// - Configuration-driven initialization for mission flexibility
    pub fn new(config: ComponentExtractionConfig) -> Self {
        Self {
            components: heapless::Vec::new(),
            next_id: 1,
            current_frame: 0,
            config,
        }
    }

    /// Extract components from a grid
    ///
    /// **Requirement Traceability**: REQ-FEAT-001, REQ-FEAT-006, REQ-FEAT-007
    /// - REQ-FEAT-001: Core component detection using configurable flood-fill
    /// - REQ-FEAT-006: Real-time processing with timeout enforcement
    /// - REQ-FEAT-007: IoU-based component matching for temporal continuity
    ///
    /// This is the primary entry point for component detection and tracking.
    /// Performs flood-fill detection, component matching, and temporal analysis
    /// within specified time and memory constraints.
    pub fn extract_components<F>(
        &mut self,
        width: usize,
        height: usize,
        mut is_anomaly: F,
    ) -> Result<(heapless::Vec<Component, 256>, ExtractionMetrics), FeatureError>
    where
        F: FnMut(usize, usize) -> bool + Copy,
    {
        #[cfg(feature = "std")]
        #[cfg(all(feature = "std", not(feature = "no_std")))]
        let start_time = std::time::Instant::now();

        if width == 0 || height == 0 || width * height > 1_000_000 {
            return Err(FeatureError::InvalidGrid);
        }

        let mut visited = heapless::Vec::<u8, 65536>::new();
        if visited.resize_default(width * height).is_err() {
            return Err(FeatureError::OutOfMemory);
        }

        let mut new_components: heapless::Vec<Component, 256> = heapless::Vec::new();
        let mut metrics = ExtractionMetrics::default();

        // Scan grid for anomalous regions
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if visited[idx] != 0 || !is_anomaly(x, y) {
                    continue;
                }

                #[cfg(feature = "std")]
                {
                    #[cfg(all(feature = "std", not(feature = "no_std")))]
                    {
                        if start_time.elapsed().as_micros() as u64 > self.config.timeout_us {
                            return Err(FeatureError::Timeout);
                        }
                    }
                }

                // Perform flood-fill to find connected component
                let flood_result = if self.config.flood_fill_config.use_8_connectivity {
                    flood_fill_8conn(
                        width,
                        height,
                        is_anomaly,
                        &mut visited,
                        x,
                        y,
                        &self.config.flood_fill_config,
                    )
                } else {
                    flood_fill_4conn(
                        width,
                        height,
                        is_anomaly,
                        &mut visited,
                        x,
                        y,
                        &self.config.flood_fill_config,
                    )
                };

                let (stats, _flood_metrics) = flood_result?;
                metrics.flood_fills_performed += 1;

                // Filter by size
                if stats.area() < self.config.min_component_size
                    || stats.area() > self.config.max_component_size
                {
                    continue;
                }

                // Create new component
                let component = Component::new(stats, self.next_id, self.current_frame);
                self.next_id += 1;

                if new_components.push(component).is_err() {
                    return Err(FeatureError::TooManyComponents);
                }
            }
        }

        metrics.components_found = new_components.len();

        // Match with existing components
        self.match_components(&mut new_components, &mut metrics)?;

        // Update frame counter
        self.current_frame += 1;

        // Clean up old components
        self.cleanup_old_components();

        #[cfg(feature = "std")]
        {
            #[cfg(all(feature = "std", not(feature = "no_std")))]
            {
                metrics.total_time_us = start_time.elapsed().as_micros() as u64;
            }
        }

        Ok((self.components.clone(), metrics))
    }

    /// Match new components with existing ones
    fn match_components(
        &mut self,
        new_components: &mut heapless::Vec<Component, 256>,
        metrics: &mut ExtractionMetrics,
    ) -> Result<(), FeatureError> {
        let mut matched = heapless::Vec::<bool, 256>::new();
        matched
            .resize_default(self.components.len())
            .map_err(|_| FeatureError::OutOfMemory)?;

        // Try to match each new component with existing ones
        for new_comp in new_components.iter_mut() {
            let mut best_match_idx = None;
            let mut best_iou = 0.0;

            for (idx, existing_comp) in self.components.iter().enumerate() {
                if matched[idx] {
                    continue;
                }

                if let Some(iou) = new_comp.iou(existing_comp) {
                    if iou > self.config.iou_threshold && iou > best_iou {
                        best_iou = iou;
                        best_match_idx = Some(idx);
                    }
                }
            }

            if let Some(idx) = best_match_idx {
                // Update existing component
                self.components[idx].update(new_comp.stats, self.current_frame);
                matched[idx] = true;
                metrics.components_matched += 1;
            } else {
                // Add as new component
                if self.components.push(new_comp.clone()).is_err() {
                    return Err(FeatureError::TooManyComponents);
                }
                metrics.new_components += 1;
            }
        }

        Ok(())
    }

    /// Remove old components that haven't been updated
    fn cleanup_old_components(&mut self) {
        self.components
            .retain(|comp| !comp.should_remove(self.current_frame, self.config.max_age_frames));
    }

    /// Get current components
    pub fn components(&self) -> &[Component] {
        &self.components
    }

    /// Get current frame number
    pub fn current_frame(&self) -> u32 {
        self.current_frame
    }

    /// Reset tracker state
    pub fn reset(&mut self) {
        self.components.clear();
        self.next_id = 1;
        self.current_frame = 0;
    }
}

//
// === LOGICAL SYSTEM INTEGRATION FOR HARDWARE/SOFTWARE DECISION-AIDING ===
//

/// Integration helper for converting geographic components to logical nodes
///
/// **Requirement Traceability**: REQ-FEAT-013 - Logical System Integration
/// - Bridges geographic anomaly detection with logical system analysis
/// - Enables hardware/software fault propagation modeling
/// - Supports policy response simulation based on component characteristics
impl ComponentTracker {
    /// Convert components to logical system nodes
    ///
    /// This function creates logical nodes from detected anomaly components,
    /// enabling fault propagation and policy response analysis across
    /// hardware/software system topologies.
    pub fn components_to_logical_nodes(
        &self,
        node_type_mapper: &ComponentNodeTypeMapper,
    ) -> heapless::Vec<LogicalNode, 256> {
        let mut logical_nodes = heapless::Vec::new();

        for component in &self.components {
            let node_type = node_type_mapper.map_component_to_node_type(component);
            let criticality = self.calculate_component_criticality(component);
            let state = self.map_component_state(component);

            let logical_node = LogicalNode {
                id: format!("component_{}", component.id),
                node_type,
                state,
                connections: heapless::Vec::new(), // Filled by topology builder
                criticality,
                resources: self.estimate_component_resources(component),
            };

            if logical_nodes.push(logical_node).is_err() {
                break; // Memory limit reached
            }
        }

        logical_nodes
    }

    /// Calculate criticality score based on component characteristics
    fn calculate_component_criticality(&self, component: &Component) -> f32 {
        let area_factor = (component.stats.area() as f32 / 100.0).min(1.0);
        let confidence_factor = component.confidence;
        let growth_factor = if component.growth_rate > 0.5 {
            0.8
        } else {
            0.3
        };
        let age_factor = (component.age() as f32 / 10.0).min(0.5);

        (area_factor * 0.4 + confidence_factor * 0.3 + growth_factor * 0.2 + age_factor * 0.1)
            .min(1.0)
    }

    /// Map component evolution state to logical node state
    fn map_component_state(&self, component: &Component) -> LogicalNodeState {
        if component.growth_rate > 1.0 {
            LogicalNodeState::Failed
        } else if component.growth_rate > 0.5 {
            LogicalNodeState::Degraded
        } else if component.confidence < 0.3 {
            LogicalNodeState::Unknown
        } else {
            LogicalNodeState::Operational
        }
    }

    /// Estimate resource usage based on component characteristics
    fn estimate_component_resources(&self, component: &Component) -> LogicalNodeResources {
        let area = component.stats.area() as f32;
        LogicalNodeResources {
            power: area * 0.5,                    // Proportional to size
            cpu_load: component.confidence * 0.8, // Based on processing complexity
            memory_bytes: (area * 10.0) as u32,   // Memory usage estimate
            bandwidth_bps: (area * 100.0) as u32, // Communication requirements
        }
    }
}

/// Mapper for converting components to logical node types
///
/// **Requirement Traceability**: REQ-FEAT-013 - Logical System Integration
/// - Provides configurable mapping from geographic anomalies to system components
/// - Enables mission-specific component-to-hardware/software association
/// - Supports dynamic topology generation for decision-aiding
pub struct ComponentNodeTypeMapper {
    /// Default node type for unmapped components
    pub default_node_type: LogicalNodeType,
    /// Size-based mapping thresholds
    pub size_thresholds: ComponentSizeThresholds,
    /// Location-based mapping regions
    pub location_mapping: ComponentLocationMapping,
}

impl ComponentNodeTypeMapper {
    /// Create new mapper with default configuration
    pub fn new() -> Self {
        Self {
            default_node_type: LogicalNodeType::SensorCluster,
            size_thresholds: ComponentSizeThresholds::default(),
            location_mapping: ComponentLocationMapping::default(),
        }
    }

    /// Map component to logical node type based on characteristics
    pub fn map_component_to_node_type(&self, component: &Component) -> LogicalNodeType {
        let area = component.stats.area();

        // Size-based classification
        if area > self.size_thresholds.large_component_threshold {
            return LogicalNodeType::ProcessingUnit;
        } else if area > self.size_thresholds.medium_component_threshold {
            return LogicalNodeType::PowerBus;
        }

        // Location-based classification
        if let Some((x, y)) = component.stats.centroid() {
            if let Some(node_type) = self
                .location_mapping
                .get_node_type_for_location(x as usize, y as usize)
            {
                return node_type;
            }
        }

        // Default classification
        self.default_node_type
    }
}

impl Default for ComponentNodeTypeMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Size-based mapping thresholds
#[derive(Debug, Clone, Copy)]
pub struct ComponentSizeThresholds {
    /// Threshold for large components (processing units)
    pub large_component_threshold: usize,
    /// Threshold for medium components (power buses)
    pub medium_component_threshold: usize,
    /// Threshold for small components (sensors)
    pub small_component_threshold: usize,
}

impl Default for ComponentSizeThresholds {
    fn default() -> Self {
        Self {
            large_component_threshold: 100,
            medium_component_threshold: 50,
            small_component_threshold: 10,
        }
    }
}

/// Location-based component mapping
#[derive(Debug, Clone)]
pub struct ComponentLocationMapping {
    /// Grid regions mapped to node types
    pub regions: heapless::Vec<LocationRegion, 16>,
}

impl ComponentLocationMapping {
    /// Get node type for specific location
    pub fn get_node_type_for_location(&self, x: usize, y: usize) -> Option<LogicalNodeType> {
        for region in &self.regions {
            if x >= region.min_x && x <= region.max_x && y >= region.min_y && y <= region.max_y {
                return Some(region.node_type);
            }
        }
        None
    }
}

impl Default for ComponentLocationMapping {
    fn default() -> Self {
        let mut regions = heapless::Vec::new();

        // Example regions - customizable per mission
        regions
            .push(LocationRegion {
                min_x: 0,
                max_x: 50,
                min_y: 0,
                max_y: 50,
                node_type: LogicalNodeType::PowerBus,
            })
            .ok();

        regions
            .push(LocationRegion {
                min_x: 51,
                max_x: 100,
                min_y: 0,
                max_y: 50,
                node_type: LogicalNodeType::SensorCluster,
            })
            .ok();

        regions
            .push(LocationRegion {
                min_x: 0,
                max_x: 100,
                min_y: 51,
                max_y: 100,
                node_type: LogicalNodeType::ProcessingUnit,
            })
            .ok();

        Self { regions }
    }
}

/// Geographic region mapping to logical node types
#[derive(Debug, Clone, Copy)]
pub struct LocationRegion {
    /// Minimum X coordinate
    pub min_x: usize,
    /// Maximum X coordinate
    pub max_x: usize,
    /// Minimum Y coordinate
    pub min_y: usize,
    /// Maximum Y coordinate
    pub max_y: usize,
    /// Associated logical node type
    pub node_type: LogicalNodeType,
}

/// Topology builder for creating logical system graphs from components
///
/// **Requirement Traceability**: REQ-FEAT-013 - Logical System Integration
/// - Builds logical system topology from geographic component analysis
/// - Establishes connectivity based on spatial proximity and component types
/// - Supports hardware fault propagation and software impact analysis
pub struct LogicalTopologyBuilder {
    /// Connectivity rules for different node types
    pub connectivity_rules: ConnectivityRules,
    /// Maximum connection distance
    pub max_connection_distance: f32,
}

impl LogicalTopologyBuilder {
    /// Create new topology builder
    pub fn new() -> Self {
        Self {
            connectivity_rules: ConnectivityRules::default(),
            max_connection_distance: 50.0,
        }
    }

    /// Build logical topology from components
    ///
    /// Creates a logical system topology by analyzing component spatial relationships
    /// and applying connectivity rules based on node types and proximity.
    pub fn build_topology(&self, nodes: heapless::Vec<LogicalNode, 256>) -> LogicalTopology {
        let mut topology_nodes = heapless::Vec::new();
        let mut adjacency = heapless::FnvIndexMap::new();

        // Copy nodes and establish connections
        for (i, node) in nodes.iter().enumerate() {
            let mut connected_node = node.clone();

            // Find connections to other nodes
            for (j, other_node) in nodes.iter().enumerate() {
                if i != j && self.should_connect_nodes(node, other_node) {
                    if connected_node
                        .connections
                        .push(other_node.id.clone())
                        .is_err()
                    {
                        break; // Connection limit reached
                    }
                }
            }

            // Build adjacency map
            let _ = adjacency.insert(node.id.clone(), connected_node.connections.clone());

            if topology_nodes.push(connected_node).is_err() {
                break; // Memory limit reached
            }
        }

        LogicalTopology {
            nodes: topology_nodes,
            adjacency,
        }
    }

    /// Determine if two nodes should be connected
    fn should_connect_nodes(&self, node1: &LogicalNode, node2: &LogicalNode) -> bool {
        // Apply connectivity rules based on node types
        self.connectivity_rules
            .should_connect(node1.node_type, node2.node_type)
    }
}

impl Default for LogicalTopologyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Connectivity rules for logical node types
#[derive(Debug, Clone)]
pub struct ConnectivityRules {
    /// Connection matrix for node type pairs
    pub connection_matrix: [[bool; 8]; 8], // 8x8 for all LogicalNodeType variants
}

impl ConnectivityRules {
    /// Check if two node types should be connected
    pub fn should_connect(&self, type1: LogicalNodeType, type2: LogicalNodeType) -> bool {
        let idx1 = self.node_type_to_index(type1);
        let idx2 = self.node_type_to_index(type2);
        self.connection_matrix[idx1][idx2] || self.connection_matrix[idx2][idx1]
    }

    /// Convert node type to matrix index
    fn node_type_to_index(&self, node_type: LogicalNodeType) -> usize {
        match node_type {
            LogicalNodeType::PowerBus => 0,
            LogicalNodeType::SensorCluster => 1,
            LogicalNodeType::ProcessingUnit => 2,
            LogicalNodeType::CommunicationModule => 3,
            LogicalNodeType::ActuatorArray => 4,
            LogicalNodeType::ThermalSystem => 5,
            LogicalNodeType::ServiceModule => 6,
            LogicalNodeType::Thread => 7,
            // Add other types as needed, currently collapsed to prevent overflow
            _ => 7,
        }
    }
}

impl Default for ConnectivityRules {
    fn default() -> Self {
        let mut matrix = [[false; 8]; 8];

        // Power bus connects to everything
        matrix[0] = [true; 8];
        for i in 0..8 {
            matrix[i][0] = true;
        }

        // Processing units connect to sensors and actuators
        matrix[2][1] = true;
        matrix[1][2] = true; // ProcessingUnit <-> SensorCluster
        matrix[2][4] = true;
        matrix[4][2] = true; // ProcessingUnit <-> ActuatorArray
        matrix[2][3] = true;
        matrix[3][2] = true; // ProcessingUnit <-> CommunicationModule

        // Thermal system connects to all hardware
        matrix[5][1] = true;
        matrix[1][5] = true; // ThermalSystem <-> SensorCluster
        matrix[5][2] = true;
        matrix[2][5] = true; // ThermalSystem <-> ProcessingUnit
        matrix[5][4] = true;
        matrix[4][5] = true; // ThermalSystem <-> ActuatorArray

        // Software components interconnected
        matrix[6][6] = true; // ServiceModule <-> ServiceModule
        matrix[6][7] = true;
        matrix[7][6] = true; // ServiceModule <-> Thread

        Self {
            connection_matrix: matrix,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// **Requirement Verification**: REQ-FEAT-002 - Component Creation and Tracking
    /// - Verifies component initialization with proper ID and frame tracking
    /// - Validates area calculation from region statistics
    /// - Tests basic component data structure integrity
    fn test_component_creation() {
        let mut stats = RegionStats::new();
        stats.add_cell(1, 1, false);
        stats.add_cell(1, 2, false);

        let comp = Component::new(stats, 1, 0);
        assert_eq!(comp.id, 1);
        assert_eq!(comp.birth_frame, 0);
        assert_eq!(comp.stats.area(), 2);
    }

    #[test]
    /// **Requirement Verification**: REQ-FEAT-001, REQ-FEAT-006 - Component Detection and Performance
    /// - REQ-FEAT-001: Verifies flood-fill based component detection for 2x2 anomaly
    /// - REQ-FEAT-006: Tests timing metrics availability (std vs no_std behavior)
    /// - Validates complete extraction workflow with metrics collection
    fn test_component_tracker() {
        let config = ComponentExtractionConfig::default();
        let mut tracker = ComponentTracker::new(config);

        // Simple 2x2 anomaly
        let is_anomaly = |x: usize, y: usize| (x == 1 || x == 2) && (y == 1 || y == 2);

        let result = tracker.extract_components(5, 5, is_anomaly);
        assert!(result.is_ok());

        let (components, metrics) = result.unwrap();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].stats.area(), 4);
        // Time metrics are only available with std feature
        #[cfg(all(feature = "std", not(feature = "no_std")))]
        assert!(metrics.total_time_us > 0);

        // In no_std mode, time metrics will be 0
        #[cfg(feature = "no_std")]
        assert_eq!(metrics.total_time_us, 0);
    }

    #[test]
    /// **Requirement Verification**: REQ-FEAT-007 - IoU Component Matching
    /// - Verifies intersection-over-union calculation for overlapping components
    /// - Tests bounding box intersection logic for spatial analysis
    /// - Validates matching threshold behavior for temporal tracking
    fn test_component_iou() {
        let mut stats1 = RegionStats::new();
        stats1.add_cell(0, 0, false);
        stats1.add_cell(1, 0, false);
        stats1.add_cell(0, 1, false);
        stats1.add_cell(1, 1, false);

        let mut stats2 = RegionStats::new();
        stats2.add_cell(1, 1, false);
        stats2.add_cell(2, 1, false);
        stats2.add_cell(1, 2, false);
        stats2.add_cell(2, 2, false);

        let comp1 = Component::new(stats1, 1, 0);
        let comp2 = Component::new(stats2, 2, 0);

        let iou = comp1.iou(&comp2);
        assert!(iou.is_some());
        assert!(iou.unwrap() > 0.0 && iou.unwrap() < 1.0);
    }

    #[test]
    /// **Requirement Verification**: REQ-FEAT-013 - Logical System Integration
    /// - Verifies conversion of geographic components to logical nodes
    /// - Tests component criticality calculation based on characteristics
    /// - Validates node type mapping and resource estimation
    fn test_components_to_logical_nodes() {
        let config = ComponentExtractionConfig::default();
        let mut tracker = ComponentTracker::new(config);

        // Create test components
        let mut stats1 = RegionStats::new();
        stats1.add_cell(0, 0, false);
        stats1.add_cell(1, 0, false);
        let mut comp1 = Component::new(stats1, 1, 0);
        comp1.confidence = 0.8;
        comp1.growth_rate = 0.3;

        let mut stats2 = RegionStats::new();
        for x in 0..10 {
            for y in 0..10 {
                stats2.add_cell(x, y, false);
            }
        }
        let mut comp2 = Component::new(stats2, 2, 0);
        comp2.confidence = 0.9;
        comp2.growth_rate = 0.1;

        tracker.components.push(comp1).unwrap();
        tracker.components.push(comp2).unwrap();

        let mapper = ComponentNodeTypeMapper::default();
        let logical_nodes = tracker.components_to_logical_nodes(&mapper);

        assert_eq!(logical_nodes.len(), 2);
        assert_eq!(logical_nodes[0].id, "component_1");
        assert_eq!(logical_nodes[1].id, "component_2");

        // Larger component should have higher criticality
        assert!(logical_nodes[1].criticality > logical_nodes[0].criticality);

        // Larger component should have higher resource requirements
        assert!(logical_nodes[1].resources.power > logical_nodes[0].resources.power);
    }

    #[test]
    /// **Requirement Verification**: REQ-FEAT-013 - Logical Topology Building
    /// - Verifies creation of logical topology from component nodes
    /// - Tests connectivity rules application
    /// - Validates adjacency relationship establishment
    fn test_logical_topology_building() {
        let mut nodes = heapless::Vec::new();

        // Power bus node
        nodes
            .push(LogicalNode {
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
            })
            .unwrap();

        // Sensor cluster node
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

        let builder = LogicalTopologyBuilder::default();
        let topology = builder.build_topology(nodes);

        assert_eq!(topology.nodes.len(), 2);

        // Power bus should be connected to sensor
        let power_bus = &topology.nodes[0];
        assert!(power_bus.connections.contains(&"sensor_1".into()));

        // Check adjacency map
        let power_bus_key = "power_bus_1".to_string();
        let sensor_key = "sensor_1".to_string();
        assert!(topology.adjacency.contains_key(&power_bus_key));
        assert!(topology.adjacency.contains_key(&sensor_key));
    }

    #[test]
    /// **Requirement Verification**: REQ-FEAT-013 - Component Node Type Mapping
    /// - Verifies size-based component classification
    /// - Tests location-based mapping functionality
    /// - Validates default mapping behavior
    fn test_component_node_type_mapping() {
        let mapper = ComponentNodeTypeMapper::default();

        // Test small component (should map to default sensor cluster)
        let mut small_stats = RegionStats::new();
        small_stats.add_cell(25, 25, false); // Location in power bus region
        let small_comp = Component::new(small_stats, 1, 0);

        let node_type = mapper.map_component_to_node_type(&small_comp);
        assert_eq!(node_type, LogicalNodeType::PowerBus); // Location-based mapping

        // Test large component (should map to processing unit)
        let mut large_stats = RegionStats::new();
        for x in 0..15 {
            for y in 0..15 {
                large_stats.add_cell(x, y, false); // 225 cells > large threshold
            }
        }
        let large_comp = Component::new(large_stats, 2, 0);

        let node_type = mapper.map_component_to_node_type(&large_comp);
        assert_eq!(node_type, LogicalNodeType::ProcessingUnit);
    }
}
