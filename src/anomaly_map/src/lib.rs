#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate alloc;

#[cfg(feature = "no_std")]
use alloc::{vec::Vec, vec};
use core::fmt;

/// Timestamp in milliseconds for temporal tracking
pub type Timestamp = u64;

/// Performance metrics for monitoring execution
#[derive(Clone, Copy, Debug)]
pub struct PerformanceMetrics {
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Memory used in bytes
    pub memory_usage_bytes: usize,
    /// Number of operations performed
    pub operations_count: usize,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            processing_time_us: 0,
            memory_usage_bytes: 0,
            operations_count: 0,
        }
    }
}

/// Error types for anomaly map operations
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyMapError {
    IndexOutOfBounds { x: usize, y: usize, width: usize, height: usize },
    InvalidDimensions { width: usize, height: usize },
    AllocationFailed { requested: usize },
    Timeout { timeout_ms: u64 },
    DataCorruption { x: usize, y: usize },
    InvalidState { reason: &'static str },
}

impl fmt::Display for AnomalyMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnomalyMapError::IndexOutOfBounds { x, y, width, height } => {
                write!(f, "Index out of bounds: ({}, {}) for grid size ({}x{})", x, y, width, height)
            }
            AnomalyMapError::InvalidDimensions { width, height } => {
                write!(f, "Invalid grid dimensions: width={}, height={}", width, height)
            }
            AnomalyMapError::AllocationFailed { requested } => {
                write!(f, "Memory allocation failed for {} bytes", requested)
            }
            AnomalyMapError::Timeout { timeout_ms } => {
                write!(f, "Grid operation timed out after {}ms", timeout_ms)
            }
            AnomalyMapError::DataCorruption { x, y } => {
                write!(f, "Data corruption detected at position ({}, {})", x, y)
            }
            AnomalyMapError::InvalidState { reason } => {
                write!(f, "Grid is in invalid state: {}", reason)
            }
        }
    }
}

#[cfg(all(feature = "std", not(feature = "no_std")))]
impl std::error::Error for AnomalyMapError {}

/// Cell state values for the anomaly grid
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    /// Normal operation state
    Normal = 0,
    /// Anomaly detected
    Anomaly = 1,
    /// Cell is being monitored
    Monitoring = 2,
    /// Cell is disabled/offline
    Disabled = 3,
}

impl From<u8> for CellState {
    fn from(value: u8) -> Self {
        match value {
            0 => CellState::Normal,
            1 => CellState::Anomaly,
            2 => CellState::Monitoring,
            3 => CellState::Disabled,
            _ => CellState::Normal, // Default to normal for invalid values
        }
    }
}

impl From<CellState> for u8 {
    fn from(state: CellState) -> Self {
        state as u8
    }
}

/// A single cell in the anomaly grid with temporal information
#[derive(Debug, Clone, Copy)]
pub struct AnomalyCell {
    /// Current state of the cell
    pub state: CellState,
    /// Timestamp when the cell was last updated
    pub last_updated: Timestamp,
    /// Confidence level (0-255)
    pub confidence: u8,
    /// Reserved for future use
    pub _reserved: u8,
}

impl Default for AnomalyCell {
    fn default() -> Self {
        Self {
            state: CellState::Normal,
            last_updated: 0,
            confidence: 255,
            _reserved: 0,
        }
    }
}

impl AnomalyCell {
    /// Create a new anomaly cell with the given state
    pub fn new(state: CellState, timestamp: Timestamp) -> Self {
        Self {
            state,
            last_updated: timestamp,
            confidence: 255,
            _reserved: 0,
        }
    }

    /// Check if the cell is in an anomalous state
    pub fn is_anomaly(&self) -> bool {
        matches!(self.state, CellState::Anomaly)
    }

    /// Check if the cell is available for operations
    pub fn is_available(&self) -> bool {
        !matches!(self.state, CellState::Disabled)
    }

    /// Update the cell state with a new timestamp
    pub fn update_state(&mut self, state: CellState, timestamp: Timestamp) {
        self.state = state;
        self.last_updated = timestamp;
    }

    /// Get the age of the cell data in milliseconds
    pub fn age(&self, current_time: Timestamp) -> u64 {
        current_time.saturating_sub(self.last_updated)
    }
}

/// Grid configuration parameters
#[derive(Debug, Clone, Copy)]
pub struct GridConfig {
    /// Grid width
    pub width: usize,
    /// Grid height
    pub height: usize,
    /// Maximum age for cell data before it's considered stale (ms)
    pub max_cell_age_ms: u64,
    /// Default confidence level for new cells
    pub default_confidence: u8,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            width: 32,
            height: 32,
            max_cell_age_ms: 5000, // 5 seconds
            default_confidence: 255,
        }
    }
}

/// Result type for grid operations
pub type GridResult<T> = Result<T, AnomalyMapError>;

/// A 2D grid for tracking satellite anomalies with temporal information
#[derive(Debug, Clone)]
pub struct AnomalyGrid {
    /// Grid configuration
    config: GridConfig,
    /// Cell data storage
    cells: Vec<AnomalyCell>,
    /// Performance metrics
    metrics: PerformanceMetrics,
}

impl AnomalyGrid {
    /// Create a new anomaly grid with the given configuration
    pub fn new(config: GridConfig) -> GridResult<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(AnomalyMapError::InvalidDimensions {
                width: config.width,
                height: config.height,
            });
        }

        let total_size = config.width * config.height;
        let data = vec![AnomalyCell::default(); total_size];

        Ok(Self {
            config,
            cells: data,
            metrics: PerformanceMetrics::default(),
        })
    }

    /// Create a new grid with default dimensions
    pub fn default() -> GridResult<Self> {
        Self::new(GridConfig::default())
    }

    /// Get grid dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.config.width, self.config.height)
    }

    /// Get the total number of cells
    pub fn cell_count(&self) -> usize {
        self.config.width * self.config.height
    }

    /// Check if coordinates are within bounds
    pub fn is_valid_coord(&self, x: usize, y: usize) -> bool {
        x < self.config.width && y < self.config.height
    }

    /// Convert 2D coordinates to linear index
    fn coord_to_index(&self, x: usize, y: usize) -> GridResult<usize> {
        if !self.is_valid_coord(x, y) {
            return Err(AnomalyMapError::IndexOutOfBounds {
                x,
                y,
                width: self.config.width,
                height: self.config.height,
            });
        }
        Ok(y * self.config.width + x)
    }

    /// Get a cell at the given coordinates
    pub fn get_cell(&self, x: usize, y: usize) -> GridResult<&AnomalyCell> {
        let index = self.coord_to_index(x, y)?;
        Ok(&self.cells[index])
    }

    /// Get a mutable cell at the given coordinates
    pub fn get_cell_mut(&mut self, x: usize, y: usize) -> GridResult<&mut AnomalyCell> {
        let index = self.coord_to_index(x, y)?;
        Ok(&mut self.cells[index])
    }

    /// Set a cell state at the given coordinates
    pub fn set_cell(&mut self, x: usize, y: usize, state: CellState, timestamp: Timestamp) -> GridResult<()> {
        let cell = self.get_cell_mut(x, y)?;
        cell.update_state(state, timestamp);
        self.metrics.operations_count += 1;
        Ok(())
    }

    /// Check if a cell is an anomaly
    pub fn is_anomaly(&self, x: usize, y: usize) -> GridResult<bool> {
        let cell = self.get_cell(x, y)?;
        Ok(cell.is_anomaly())
    }

    /// Get all neighboring coordinates (4-connectivity)
    pub fn get_neighbors_4(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let mut neighbors = Vec::new();

        // North
        if y > 0 {
            neighbors.push((x, y - 1));
        }
        // East
        if x + 1 < self.config.width {
            neighbors.push((x + 1, y));
        }
        // South
        if y + 1 < self.config.height {
            neighbors.push((x, y + 1));
        }
        // West
        if x > 0 {
            neighbors.push((x - 1, y));
        }

        neighbors
    }

    /// Get all neighboring coordinates (8-connectivity)
    pub fn get_neighbors_8(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let mut neighbors = Vec::new();

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue; // Skip the center cell
                }

                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && ny >= 0 {
                    let ux = nx as usize;
                    let uy = ny as usize;
                    if ux < self.config.width && uy < self.config.height {
                        neighbors.push((ux, uy));
                    }
                }
            }
        }

        neighbors
    }

    /// Count anomaly cells in the grid
    pub fn count_anomalies(&self) -> usize {
        self.cells.iter().filter(|cell| cell.is_anomaly()).count()
    }

    /// Clear all cells to normal state
    pub fn clear(&mut self, timestamp: Timestamp) {
        for cell in &mut self.cells {
            cell.update_state(CellState::Normal, timestamp);
        }
        self.metrics.operations_count += 1;
    }

    /// Get performance metrics
    pub fn metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    /// Update performance metrics
    pub fn update_metrics(&mut self, metrics: PerformanceMetrics) {
        self.metrics = metrics;
    }

    /// Check for stale cells that haven't been updated recently
    pub fn find_stale_cells(&self, current_time: Timestamp) -> Vec<(usize, usize)> {
        let mut stale_cells = Vec::new();

        for y in 0..self.config.height {
            for x in 0..self.config.width {
                let cell = &self.cells[y * self.config.width + x];
                if cell.age(current_time) > self.config.max_cell_age_ms {
                    stale_cells.push((x, y));
                }
            }
        }

        stale_cells
    }

    /// Create an iterator over all cells with their coordinates
    pub fn iter_cells(&self) -> impl Iterator<Item = (usize, usize, &AnomalyCell)> {
        self.cells.iter().enumerate().map(move |(index, cell)| {
            let x = index % self.config.width;
            let y = index / self.config.width;
            (x, y, cell)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_state_conversion() {
        assert_eq!(CellState::from(0), CellState::Normal);
        assert_eq!(CellState::from(1), CellState::Anomaly);
        assert_eq!(u8::from(CellState::Anomaly), 1);
    }

    #[test]
    fn test_anomaly_cell_creation() {
        let cell = AnomalyCell::new(CellState::Anomaly, 1000);
        assert!(cell.is_anomaly());
        assert!(cell.is_available());
        assert_eq!(cell.age(1500), 500);
    }

    #[test]
    fn test_grid_creation() {
        let config = GridConfig {
            width: 10,
            height: 10,
            max_cell_age_ms: 1000,
            default_confidence: 200,
        };

        let grid = AnomalyGrid::new(config).unwrap();
        assert_eq!(grid.dimensions(), (10, 10));
        assert_eq!(grid.cell_count(), 100);
    }

    #[test]
    fn test_grid_bounds_checking() {
        let grid = AnomalyGrid::default().unwrap();
        let (width, height) = grid.dimensions();

        assert!(grid.is_valid_coord(0, 0));
        assert!(grid.is_valid_coord(width - 1, height - 1));
        assert!(!grid.is_valid_coord(width, height));
    }

    #[test]
    fn test_cell_operations() {
        let mut grid = AnomalyGrid::default().unwrap();

        // Set a cell to anomaly state
        grid.set_cell(5, 5, CellState::Anomaly, 1000).unwrap();
        assert!(grid.is_anomaly(5, 5).unwrap());

        // Count anomalies
        assert_eq!(grid.count_anomalies(), 1);

        // Clear grid
        grid.clear(2000);
        assert_eq!(grid.count_anomalies(), 0);
    }

    #[test]
    fn test_neighbors() {
        let grid = AnomalyGrid::default().unwrap();

        // Test 4-connectivity
        let neighbors_4 = grid.get_neighbors_4(5, 5);
        assert_eq!(neighbors_4.len(), 4);

        // Test 8-connectivity
        let neighbors_8 = grid.get_neighbors_8(5, 5);
        assert_eq!(neighbors_8.len(), 8);

        // Test corner cell (fewer neighbors)
        let corner_neighbors = grid.get_neighbors_4(0, 0);
        assert_eq!(corner_neighbors.len(), 2);
    }
}
