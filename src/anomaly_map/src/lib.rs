#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate alloc;

#[cfg(feature = "no_std")]
use alloc::vec::Vec;
use core::fmt;
use thiserror::Error;

/// Timestamp in milliseconds for temporal tracking
pub type Timestamp = u64;

/// Performance metrics for monitoring execution
#[derive(Clone, Copy, Debug)]
pub struct PerformanceMetrics {
    pub execution_time_us: u64,
    pub memory_used_bytes: usize,
    pub cells_processed: usize,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_time_us: 0,
            memory_used_bytes: 0,
            cells_processed: 0,
        }
    }
}

/// Error types for anomaly map operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AnomalyMapError {
    #[error("Index out of bounds: ({x}, {y}) for grid size ({width}x{height})")]
    IndexOutOfBounds { x: usize, y: usize, width: usize, height: usize },
    
    #[error("Invalid grid dimensions: width={width}, height={height}")]
    InvalidDimensions { width: usize, height: usize },
    
    #[error("Memory allocation failed for {requested} bytes")]
    MemoryAllocationFailed { requested: usize },
    
    #[error("Grid operation timed out after {timeout_ms}ms")]
    OperationTimeout { timeout_ms: u64 },
    
    #[error("Data corruption detected at position ({x}, {y})")]
    DataCorruption { x: usize, y: usize },
    
    #[error("Grid is in invalid state: {reason}")]
    InvalidState { reason: &'static str },
}

/// Result type for anomaly map operations
pub type Result<T> = core::result::Result<T, AnomalyMapError>;

/// Cell status with enhanced states for better fault detection
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellStatus {
    /// Normal operating conditions
    Nominal,
    /// Anomalous conditions detected
    Anomalous,
    /// Cell is offline/non-responsive
    Offline,
    /// Cell is under maintenance/calibration
    Maintenance,
    /// Cell data is suspect/unverified
    Suspect,
}

impl Default for CellStatus {
    fn default() -> Self {
        CellStatus::Nominal
    }
}

impl fmt::Display for CellStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CellStatus::Nominal => write!(f, "NOMINAL"),
            CellStatus::Anomalous => write!(f, "ANOMALOUS"),
            CellStatus::Offline => write!(f, "OFFLINE"),
            CellStatus::Maintenance => write!(f, "MAINTENANCE"),
            CellStatus::Suspect => write!(f, "SUSPECT"),
        }
    }
}

/// Enhanced cell structure with temporal tracking and bounds checking
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub status: CellStatus,
    /// Temperature in tenths of degrees Celsius (-400.0 to +850.0°C range)
    pub temp_dc: i16,
    /// Current in milliamps
    pub current_ma: i16,
    /// Radiation counts per second
    pub radiation_cps: u16,
    /// Timestamp of last update
    pub last_update: Timestamp,
    /// Number of anomaly detections for this cell
    pub anomaly_count: u16,
    /// Cell health metric (0-100%)
    pub health_percent: u8,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            status: CellStatus::Nominal,
            temp_dc: 200,  // 20.0°C
            current_ma: 0,
            radiation_cps: 0,
            last_update: 0,
            anomaly_count: 0,
            health_percent: 100,
        }
    }
}

impl Cell {
    /// Create a new cell with specified values and bounds checking
    pub fn new(
        temp_dc: i16,
        current_ma: i16,
        radiation_cps: u16,
        timestamp: Timestamp,
    ) -> Result<Self> {
        // Temperature bounds: -40.0°C to +85.0°C for space electronics
        if temp_dc < -400 || temp_dc > 850 {
            return Err(AnomalyMapError::InvalidState {
                reason: "Temperature out of valid range"
            });
        }
        
        // Current bounds: reasonable for small satellite components
        if current_ma < -10000 || current_ma > 10000 {
            return Err(AnomalyMapError::InvalidState {
                reason: "Current out of valid range"
            });
        }
        
        Ok(Self {
            status: CellStatus::Nominal,
            temp_dc,
            current_ma,
            radiation_cps,
            last_update: timestamp,
            anomaly_count: 0,
            health_percent: 100,
        })
    }
    
    /// Update cell values with bounds checking and anomaly detection
    pub fn update(
        &mut self,
        temp_dc: i16,
        current_ma: i16,
        radiation_cps: u16,
        timestamp: Timestamp,
    ) -> Result<()> {
        // Validate inputs
        if temp_dc < -400 || temp_dc > 850 {
            self.status = CellStatus::Suspect;
            return Err(AnomalyMapError::InvalidState {
                reason: "Temperature reading out of range"
            });
        }
        
        if current_ma < -10000 || current_ma > 10000 {
            self.status = CellStatus::Suspect;
            return Err(AnomalyMapError::InvalidState {
                reason: "Current reading out of range"
            });
        }
        
        // Check for temporal consistency
        if timestamp < self.last_update {
            self.status = CellStatus::Suspect;
            return Err(AnomalyMapError::InvalidState {
                reason: "Timestamp regression detected"
            });
        }
        
        // Update values
        let old_temp = self.temp_dc;
        self.temp_dc = temp_dc;
        self.current_ma = current_ma;
        self.radiation_cps = radiation_cps;
        self.last_update = timestamp;
        
        // Simple anomaly detection based on temperature delta
        let temp_delta = (temp_dc - old_temp).abs();
        if temp_delta > 100 {  // 10°C sudden change
            self.anomaly_count = self.anomaly_count.saturating_add(1);
            self.status = CellStatus::Anomalous;
            self.health_percent = self.health_percent.saturating_sub(5);
        } else if self.status == CellStatus::Anomalous && temp_delta < 20 {
            // Recovery condition
            self.status = CellStatus::Nominal;
            self.health_percent = (self.health_percent + 1).min(100);
        }
        
        // High radiation detection
        if radiation_cps > 1000 {
            self.anomaly_count = self.anomaly_count.saturating_add(1);
            self.status = CellStatus::Anomalous;
            self.health_percent = self.health_percent.saturating_sub(2);
        }
        
        Ok(())
    }
    
    /// Get temperature in degrees Celsius
    pub fn temp_celsius(&self) -> f32 {
        self.temp_dc as f32 / 10.0
    }
    
    /// Check if cell is considered healthy
    pub fn is_healthy(&self) -> bool {
        self.health_percent > 70 && self.status != CellStatus::Offline
    }
}

/// Enhanced 2D grid with comprehensive error handling and performance monitoring
pub struct Grid2D<T> {
    pub width: usize,
    pub height: usize,
    data: Vec<T>,
    /// Creation timestamp
    pub created_at: Timestamp,
    /// Last modification timestamp
    pub last_modified: Timestamp,
    /// Total number of updates performed
    pub update_count: u64,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

impl<T: Copy + Default> Grid2D<T> {
    /// Create a new grid with bounds checking and memory management
    pub fn new(width: usize, height: usize, init: T) -> Result<Self> {
        // Validate dimensions
        if width == 0 || height == 0 {
            return Err(AnomalyMapError::InvalidDimensions { width, height });
        }
        
        // Check for overflow
        let total_size = width.checked_mul(height)
            .ok_or(AnomalyMapError::InvalidDimensions { width, height })?;
        
        // Reasonable size limit for embedded systems (64KB default)
        const MAX_CELLS: usize = 65536;
        if total_size > MAX_CELLS {
            return Err(AnomalyMapError::MemoryAllocationFailed { 
                requested: total_size * core::mem::size_of::<T>() 
            });
        }
        
        let data = vec![init; total_size];
        let now = get_timestamp();
        
        Ok(Self {
            width,
            height,
            data,
            created_at: now,
            last_modified: now,
            update_count: 0,
            metrics: PerformanceMetrics::default(),
        })
    }
    
    /// Create a grid with default values
    pub fn new_default(width: usize, height: usize) -> Result<Self> {
        Self::new(width, height, T::default())
    }
    
    /// Get linear index with bounds checking
    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> Result<usize> {
        if x >= self.width || y >= self.height {
            return Err(AnomalyMapError::IndexOutOfBounds {
                x, y, width: self.width, height: self.height
            });
        }
        Ok(y * self.width + x)
    }
    
    /// Check if coordinates are within bounds
    #[inline]
    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }
    
    /// Safe coordinate checking for usize
    #[inline]
    pub fn in_bounds_usize(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }
    
    /// Get cell reference with bounds checking
    pub fn get(&self, x: usize, y: usize) -> Result<&T> {
        let idx = self.idx(x, y)?;
        Ok(&self.data[idx])
    }
    
    /// Get mutable cell reference with bounds checking
    pub fn get_mut(&mut self, x: usize, y: usize) -> Result<&mut T> {
        let idx = self.idx(x, y)?;
        self.last_modified = get_timestamp();
        self.update_count += 1;
        Ok(&mut self.data[idx])
    }
    
    /// Safely set cell value
    pub fn set(&mut self, x: usize, y: usize, value: T) -> Result<()> {
        let idx = self.idx(x, y)?;
        self.data[idx] = value;
        self.last_modified = get_timestamp();
        self.update_count += 1;
        Ok(())
    }
    
    /// Get grid dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
    
    /// Get total number of cells
    pub fn cell_count(&self) -> usize {
        self.width * self.height
    }
    
    /// Clear all cells to default value
    pub fn clear(&mut self) {
        for cell in &mut self.data {
            *cell = T::default();
        }
        self.last_modified = get_timestamp();
        self.update_count += 1;
    }
    
    /// Get iterator over all cells
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
    
    /// Get mutable iterator over all cells
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.last_modified = get_timestamp();
        self.data.iter_mut()
    }
    
    /// Validate grid integrity (detect corruption)
    pub fn validate(&self) -> Result<()> {
        if self.data.len() != self.width * self.height {
            return Err(AnomalyMapError::DataCorruption { x: 0, y: 0 });
        }
        
        if self.width == 0 || self.height == 0 {
            return Err(AnomalyMapError::InvalidState {
                reason: "Grid has zero dimensions"
            });
        }
        
        Ok(())
    }
}

/// Get current timestamp in milliseconds (placeholder for actual time source)
pub fn get_timestamp() -> Timestamp {
    // In real implementation, this would interface with spacecraft time
    // For now, use a simple counter or system time
    static mut COUNTER: u64 = 0;
    unsafe {
        COUNTER += 1;
        COUNTER
    }
}
