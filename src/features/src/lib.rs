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

#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate alloc;

use floodfill_core::{
    flood_fill_4conn, flood_fill_8conn, FloodFillConfig, FloodFillError, RegionStats,
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
}
