#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate alloc;

use floodfill_core::{
    flood_fill_4conn, flood_fill_8conn, FloodFillConfig, FloodFillError, FloodFillMetrics,
    FloodFillResult, RegionStats,
};

#[cfg(all(feature = "std", not(feature = "no_std")))]
use std::time::Instant;

/// Threat level for anomaly classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThreatLevel {
    None = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl Default for ThreatLevel {
    fn default() -> Self {
        ThreatLevel::None
    }
}

/// Evolution state for temporal analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvolutionState {
    Stable,
    Growing,
    Shrinking,
    Oscillating,
    Disappeared,
}

impl Default for EvolutionState {
    fn default() -> Self {
        EvolutionState::Stable
    }
}

/// Legacy Feature type for backward compatibility with decisions crate
pub type Feature = Component;

/// Legacy FeatureTracker type for backward compatibility
pub type FeatureTracker = ComponentTracker;

/// Error types for feature extraction operations
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

/// A connected component with statistics and temporal tracking
#[derive(Clone, Debug)]
pub struct Component {
    /// Region statistics
    pub stats: RegionStats,
    /// Component ID for tracking across frames
    pub id: u32,
    /// Frame number when first detected
    pub birth_frame: u32,
    /// Frame number when last seen
    pub last_seen_frame: u32,
    /// Growth rate in cells per frame
    pub growth_rate: f32,
    /// Previous area for growth calculation
    pub previous_area: usize,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Whether this component is stable
    pub is_stable: bool,
}

impl Component {
    /// Create a new component
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
    pub fn age(&self) -> u32 {
        self.last_seen_frame.saturating_sub(self.birth_frame)
    }

    /// Check if component should be considered for removal
    pub fn should_remove(&self, current_frame: u32, max_age_without_update: u32) -> bool {
        current_frame.saturating_sub(self.last_seen_frame) > max_age_without_update
    }

    /// Calculate intersection over union with another component
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
#[derive(Debug, Clone, Copy)]
pub struct ComponentExtractionConfig {
    /// Maximum number of components to track
    pub max_components: usize,
    /// Minimum component size in cells
    pub min_component_size: usize,
    /// Maximum component size in cells
    pub max_component_size: usize,
    /// IoU threshold for component matching
    pub iou_threshold: f32,
    /// Maximum age without update before removal
    pub max_age_frames: u32,
    /// Processing timeout in microseconds
    pub timeout_us: u64,
    /// Flood-fill configuration
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
#[derive(Debug, Clone, Copy, Default)]
pub struct ExtractionMetrics {
    /// Total processing time in microseconds
    pub total_time_us: u64,
    /// Number of flood-fill operations performed
    pub flood_fills_performed: usize,
    /// Number of components found
    pub components_found: usize,
    /// Number of components matched with previous frame
    pub components_matched: usize,
    /// Number of new components
    pub new_components: usize,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
}

/// Component tracker for temporal analysis
pub struct ComponentTracker {
    /// Current components
    components: heapless::Vec<Component, 256>,
    /// Next component ID
    next_id: u32,
    /// Current frame number
    current_frame: u32,
    /// Configuration
    config: ComponentExtractionConfig,
}

impl ComponentTracker {
    /// Create a new component tracker
    pub fn new(config: ComponentExtractionConfig) -> Self {
        Self {
            components: heapless::Vec::new(),
            next_id: 1,
            current_frame: 0,
            config,
        }
    }

    /// Extract components from a grid
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
        assert!(metrics.total_time_us > 0);
    }

    #[test]
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
