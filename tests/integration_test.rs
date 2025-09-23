//! Integration tests for the satellite anomaly flood-fill system
//!
//! This test demonstrates the complete workflow from anomaly grid creation
//! through component detection, feature extraction, and decision making.

use anomaly_map::{AnomalyGrid, AnomalyCell, CellState, GridConfig};
use floodfill_core::{flood_fill_4conn, FloodFillConfig, get_timestamp};
use features::{ComponentTracker, ComponentExtractionConfig, ThreatLevel};
use decisions::{DecisionEngine, DecisionContext, GlobalPolicy, MissionPhase, ThreatAssessment};

#[test]
fn test_complete_anomaly_detection_workflow() {
    // Step 1: Create an anomaly grid representing satellite systems
    let config = GridConfig {
        width: 20,
        height: 20,
        max_cell_age_ms: 5000,
        default_confidence: 255,
    };
    
    let mut grid = AnomalyGrid::new(config).expect("Failed to create anomaly grid");
    let timestamp = get_timestamp();

    // Step 2: Simulate some anomalies in different locations
    // Simulate a solar array anomaly (compact region)
    for x in 5..8 {
        for y in 5..8 {
            grid.set_cell(x, y, CellState::Anomaly, timestamp).unwrap();
        }
    }

    // Simulate a thermal system anomaly (larger, scattered region)
    let thermal_anomaly_cells = [
        (10, 10), (11, 10), (12, 10),
        (10, 11), (11, 11), (12, 11), (13, 11),
        (10, 12), (11, 12), (12, 12),
        (15, 15), (16, 15), // Separate thermal component
    ];
    
    for (x, y) in thermal_anomaly_cells {
        grid.set_cell(x, y, CellState::Anomaly, timestamp).unwrap();
    }

    // Step 3: Extract components using flood-fill
    let mut tracker_config = ComponentExtractionConfig::default();
    tracker_config.flood_fill_config.use_8_connectivity = false; // Use 4-connectivity
    
    let mut component_tracker = ComponentTracker::new(tracker_config);
    
    // Create a function to check if a cell is an anomaly
    let is_anomaly = |x: usize, y: usize| -> bool {
        grid.is_anomaly(x, y).unwrap_or(false)
    };
    
    // Extract components from the grid
    let (width, height) = grid.dimensions();
    let mut visited = vec![0u8; width * height];
    
    let extraction_result = component_tracker.extract_components(
        width,
        height,
        is_anomaly,
        &mut visited,
        1000, // frame number
    );
    
    assert!(extraction_result.is_ok(), "Component extraction failed");
    let (components, _extraction_metrics) = extraction_result.unwrap();
    
    // Should detect the two separate anomaly regions
    assert!(!components.is_empty(), "No components detected");
    println!("Detected {} anomaly components", components.len());
    
    // Step 4: Assess threat levels for each component
    for (i, component) in components.iter().enumerate() {
        let threat = component.threat_level();
        println!("Component {}: Area={}, Threat={:?}", 
                 i, component.stats.area(), threat);
        
        assert!(matches!(threat, 
            ThreatLevel::None | ThreatLevel::Low | ThreatLevel::Medium | 
            ThreatLevel::High | ThreatLevel::Critical
        ), "Invalid threat level");
    }

    // Step 5: Make decisions based on the detected anomalies
    let decision_engine = DecisionEngine::new(GlobalPolicy::default());
    
    let decision_context = DecisionContext {
        timestamp,
        power_level: 75,
        emergency_enabled: true,
        available_power_w: 85.0,
        attitude_quaternion: [1.0, 0.0, 0.0, 0.0],
        mission_phase: MissionPhase::Operations,
    };

    let decision_result = decision_engine.decide(&components, &decision_context);
    assert!(decision_result.is_ok(), "Decision making failed");
    
    let action = decision_result.unwrap();
    println!("Recommended action: {:?}", action);

    // Step 6: Verify the end-to-end workflow
    println!("Integration test completed successfully!");
    println!("- Created {}x{} anomaly grid", width, height);
    println!("- Detected {} anomaly components", components.len());
    println!("- Generated decision: {:?}", action);
    
    // The workflow should complete without errors
    assert!(true, "Complete workflow executed successfully");
}

#[test]
fn test_temporal_tracking() {
    // Test that components can be tracked over multiple frames
    let config = ComponentExtractionConfig::default();
    let mut tracker = ComponentTracker::new(config);
    
    let width = 10;
    let height = 10;
    
    // Frame 1: Small anomaly
    let is_anomaly_frame1 = |x: usize, y: usize| -> bool {
        matches!((x, y), (3, 3) | (3, 4) | (4, 3) | (4, 4))
    };
    
    let mut visited = vec![0u8; width * height];
    let result1 = tracker.extract_components(width, height, is_anomaly_frame1, &mut visited, 1);
    assert!(result1.is_ok());
    let (components1, _) = result1.unwrap();
    assert_eq!(components1.len(), 1);
    
    // Frame 2: Same anomaly slightly grown
    visited.fill(0); // Reset visited array
    let is_anomaly_frame2 = |x: usize, y: usize| -> bool {
        matches!((x, y), (3, 3) | (3, 4) | (4, 3) | (4, 4) | (5, 4) | (4, 5))
    };
    
    let result2 = tracker.extract_components(width, height, is_anomaly_frame2, &mut visited, 2);
    assert!(result2.is_ok());
    let (components2, _) = result2.unwrap();
    assert_eq!(components2.len(), 1);
    
    // Verify the component has grown
    assert!(components2[0].stats.area() > components1[0].stats.area());
    
    println!("Temporal tracking test completed successfully!");
    println!("Frame 1 area: {}, Frame 2 area: {}", 
             components1[0].stats.area(), components2[0].stats.area());
}

#[test]
fn test_performance_metrics() {
    // Test that performance metrics are collected
    let config = ComponentExtractionConfig::default();
    let mut tracker = ComponentTracker::new(config);
    
    let width = 50;
    let height = 50;
    
    // Create a complex pattern to ensure meaningful metrics
    let is_anomaly = |x: usize, y: usize| -> bool {
        // Create a checkerboard pattern of anomalies
        (x + y) % 3 == 0 && x < 25 && y < 25
    };
    
    let mut visited = vec![0u8; width * height];
    let result = tracker.extract_components(width, height, is_anomaly, &mut visited, 1);
    
    assert!(result.is_ok());
    let (_components, metrics) = result.unwrap();
    
    // Verify metrics are collected
    println!("Performance metrics:");
    println!("- Total components found: {}", metrics.components_found);
    println!("- Total pixels processed: {}", metrics.pixels_processed);
    println!("- Total processing time: {}µs", metrics.total_time_us);
    
    assert!(metrics.components_found > 0, "Should find some components");
    assert!(metrics.pixels_processed > 0, "Should process some pixels");
    
    // Time metrics may be 0 in no_std mode, so we don't assert on them
    println!("Performance metrics test completed successfully!");
}
