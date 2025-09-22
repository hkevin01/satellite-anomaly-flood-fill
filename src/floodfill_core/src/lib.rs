#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate alloc;

use core::fmt;
use thiserror::Error;

#[cfg(feature = "no_std")]
use alloc::vec::Vec;

/// Timestamp type for performance tracking
pub type Timestamp = u64;

/// Error types for flood-fill operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum FloodFillError {
    #[error("Stack overflow: region too large (>{max_size} cells)")]
    StackOverflow { max_size: usize },
    
    #[error("Invalid grid dimensions: {width}x{height}")]
    InvalidDimensions { width: usize, height: usize },
    
    #[error("Start position ({x}, {y}) out of bounds for {width}x{height} grid")]
    StartOutOfBounds { x: usize, y: usize, width: usize, height: usize },
    
    #[error("Operation timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
    
    #[error("Memory allocation failed")]
    MemoryError,
    
    #[error("Invalid visited buffer size: expected {expected}, got {actual}")]
    InvalidVisitedSize { expected: usize, actual: usize },
}

/// Result type for flood-fill operations
pub type Result<T> = core::result::Result<T, FloodFillError>;

/// Performance metrics for flood-fill operations
#[derive(Clone, Copy, Debug, Default)]
pub struct FloodFillMetrics {
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Number of cells processed
    pub cells_processed: usize,
    /// Stack usage high watermark
    pub max_stack_depth: usize,
    /// Number of boundary checks performed
    pub boundary_checks: u64,
    /// Memory bytes used for operation
    pub memory_used: usize,
}

/// Enhanced region statistics with comprehensive metrics
#[derive(Clone, Copy, Debug)]
pub struct RegionStats {
    /// Number of cells in the region
    pub count: usize,
    /// Sum of X coordinates for centroid calculation
    pub sum_x: u64,
    /// Sum of Y coordinates for centroid calculation  
    pub sum_y: u64,
    /// Minimum X coordinate (bounding box)
    pub min_x: usize,
    /// Minimum Y coordinate (bounding box)
    pub min_y: usize,
    /// Maximum X coordinate (bounding box)
    pub max_x: usize,
    /// Maximum Y coordinate (bounding box)
    pub max_y: usize,
    /// Perimeter length (approximate)
    pub perimeter: usize,
    /// Timestamp when region was detected
    pub detection_time: Timestamp,
    /// Performance metrics for this fill
    pub metrics: FloodFillMetrics,
}

impl Default for RegionStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RegionStats {
    /// Create new empty region statistics
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
            detection_time: get_timestamp(),
            metrics: FloodFillMetrics::default(),
        }
    }
    
    /// Add a cell to the region statistics with overflow protection
    pub fn add(&mut self, x: usize, y: usize) -> Result<()> {
        // Check for potential overflow
        if self.count == usize::MAX {
            return Err(FloodFillError::StackOverflow { max_size: usize::MAX });
        }
        
        self.count += 1;
        
        // Use saturating arithmetic to prevent overflow
        self.sum_x = self.sum_x.saturating_add(x as u64);
        self.sum_y = self.sum_y.saturating_add(y as u64);
        
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        
        Ok(())
    }
    
    /// Calculate region centroid with error handling
    pub fn centroid(&self) -> Option<(f32, f32)> {
        if self.count == 0 {
            None
        } else {
            Some((
                self.sum_x as f32 / self.count as f32,
                self.sum_y as f32 / self.count as f32
            ))
        }
    }
    
    /// Get bounding box dimensions
    pub fn bounding_box(&self) -> Option<(usize, usize, usize, usize)> {
        if self.count == 0 {
            None
        } else {
            Some((self.min_x, self.min_y, self.max_x, self.max_y))
        }
    }
    
    /// Calculate bounding box area
    pub fn bounding_area(&self) -> usize {
        if self.count == 0 {
            0
        } else {
            (self.max_x - self.min_x + 1) * (self.max_y - self.min_y + 1)
        }
    }
    
    /// Calculate aspect ratio of bounding box
    pub fn aspect_ratio(&self) -> Option<f32> {
        if self.count == 0 {
            None
        } else {
            let width = (self.max_x - self.min_x + 1) as f32;
            let height = (self.max_y - self.min_y + 1) as f32;
            Some(width / height.max(1.0))
        }
    }
    
    /// Calculate fill density (region cells / bounding box area)
    pub fn density(&self) -> f32 {
        let bbox_area = self.bounding_area();
        if bbox_area == 0 {
            0.0
        } else {
            self.count as f32 / bbox_area as f32
        }
    }
    
    /// Check if region is compact (high density, low aspect ratio)
    pub fn is_compact(&self) -> bool {
        let density = self.density();
        let aspect = self.aspect_ratio().unwrap_or(1.0);
        density > 0.5 && aspect >= 0.5 && aspect <= 2.0
    }
}

impl fmt::Display for RegionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some((cx, cy)) = self.centroid() {
            write!(f, "Region: {} cells, centroid ({:.1}, {:.1}), bbox ({}, {}) to ({}, {})",
                   self.count, cx, cy, self.min_x, self.min_y, self.max_x, self.max_y)
        } else {
            write!(f, "Empty region")
        }
    }
}

/// Connectivity type for flood-fill algorithms
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Connectivity {
    /// 4-connected (N, E, S, W neighbors)
    Four,
    /// 8-connected (including diagonals)
    Eight,
}

/// Enhanced flood-fill with 4-connectivity
pub fn flood_fill_4conn<F>(
    width: usize,
    height: usize,
    is_anom: F,
    visited: &mut [u8],
    start_x: usize,
    start_y: usize,
) -> Result<RegionStats>
where
    F: FnMut(usize, usize) -> bool,
{
    flood_fill_with_connectivity(
        width, height, is_anom, visited, start_x, start_y, Connectivity::Four
    )
}

/// Enhanced flood-fill with 8-connectivity
pub fn flood_fill_8conn<F>(
    width: usize,
    height: usize,
    is_anom: F,
    visited: &mut [u8],
    start_x: usize,
    start_y: usize,
) -> Result<RegionStats>
where
    F: FnMut(usize, usize) -> bool,
{
    flood_fill_with_connectivity(
        width, height, is_anom, visited, start_x, start_y, Connectivity::Eight
    )
}

/// Generic flood-fill implementation with configurable connectivity
pub fn flood_fill_with_connectivity<F>(
    width: usize,
    height: usize,
    mut is_anom: F,
    visited: &mut [u8],
    start_x: usize,
    start_y: usize,
    connectivity: Connectivity,
) -> Result<RegionStats>
where
    F: FnMut(usize, usize) -> bool,
{
    let start_time = get_timestamp();
    let mut metrics = FloodFillMetrics::default();
    
    // Input validation
    if width == 0 || height == 0 {
        return Err(FloodFillError::InvalidDimensions { width, height });
    }
    
    if start_x >= width || start_y >= height {
        return Err(FloodFillError::StartOutOfBounds { 
            x: start_x, y: start_y, width, height 
        });
    }
    
    let grid_size = width * height;
    if visited.len() != grid_size {
        return Err(FloodFillError::InvalidVisitedSize {
            expected: grid_size,
            actual: visited.len()
        });
    }
    
    // Simple stack for no_std compatibility
    let mut stack = Vec::new();
    let mut stats = RegionStats::new();
    
    let start_idx = start_y * width + start_x;
    
    // Check if starting point is valid and not already visited
    if visited[start_idx] != 0 || !is_anom(start_x, start_y) {
        stats.metrics = metrics;
        return Ok(stats);
    }
    
    // Initialize flood-fill
    stack.push((start_x, start_y));
    visited[start_idx] = 1;
    metrics.cells_processed += 1;
    
    // Main flood-fill loop with timeout protection
    const MAX_ITERATIONS: usize = 100_000;
    let mut iterations = 0;
    
    while let Some((x, y)) = stack.pop() {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            return Err(FloodFillError::Timeout { timeout_ms: 1000 });
        }
        
        // Add current cell to region
        stats.add(x, y)?;
        
        // Track stack depth
        metrics.max_stack_depth = metrics.max_stack_depth.max(stack.len());
        
        // Get neighbors based on connectivity
        let neighbors = match connectivity {
            Connectivity::Four => get_4_neighbors(x as i32, y as i32),
            Connectivity::Eight => get_8_neighbors(x as i32, y as i32),
        };
        
        // Process each neighbor
        for (nx, ny) in neighbors {
            metrics.boundary_checks += 1;
            
            // Bounds checking
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
            
            // Check if neighbor is anomalous
            if !is_anom(ux, uy) {
                continue;
            }
            
            // Mark as visited and add to stack
            visited[nidx] = 1;
            metrics.cells_processed += 1;
            
            stack.push((ux, uy));
        }
    }
    
    // Calculate execution time
    let end_time = get_timestamp();
    metrics.execution_time_us = end_time.saturating_sub(start_time);
    metrics.memory_used = stack.capacity() * core::mem::size_of::<(usize, usize)>();
    
    stats.metrics = metrics;
    Ok(stats)
}

/// Get 4-connected neighbors (N, E, S, W)
fn get_4_neighbors(x: i32, y: i32) -> Vec<(i32, i32)> {
    vec![
        (x, y - 1),     // North
        (x + 1, y),     // East
        (x, y + 1),     // South
        (x - 1, y),     // West
    ]
}

/// Get 8-connected neighbors (including diagonals)
fn get_8_neighbors(x: i32, y: i32) -> Vec<(i32, i32)> {
    vec![
        (x - 1, y - 1), // NW
        (x, y - 1),     // N
        (x + 1, y - 1), // NE
        (x + 1, y),     // E
        (x + 1, y + 1), // SE
        (x, y + 1),     // S
        (x - 1, y + 1), // SW
        (x - 1, y),     // W
    ]
}

/// Batch flood-fill for finding all connected components in a grid
pub fn find_all_components<F>(
    width: usize,
    height: usize,
    mut is_anom: F,
    connectivity: Connectivity,
) -> Result<Vec<RegionStats>>
where
    F: FnMut(usize, usize) -> bool + Copy,
{
    let grid_size = width * height;
    let mut visited = vec![0u8; grid_size];
    let mut components = Vec::new();
    
    // Iterate through all cells to find unvisited anomalous regions
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            
            // Skip if already visited or not anomalous
            if visited[idx] != 0 || !is_anom(x, y) {
                continue;
            }
            
            // Found new component - perform flood-fill
            let stats = flood_fill_with_connectivity(
                width, height, is_anom, &mut visited, x, y, connectivity
            )?;
            
            // Only add non-empty components
            if stats.count > 0 {
                components.push(stats);
            }
        }
    }
    
    Ok(components)
}

/// Get current timestamp (placeholder implementation)
pub fn get_timestamp() -> Timestamp {
    // In real spacecraft implementation, this would interface with mission time
    // For now, use a simple monotonic counter
    static mut COUNTER: Timestamp = 0;
    unsafe {
        COUNTER += 1;
        COUNTER
    }
}
