# -*- coding: utf-8 -*-
"""
GPR B-Scan Visualization and Analysis System
Comprehensive visualization for Ground Penetrating Radar data including:
- B-scan profile visualization
- 3D GPR data visualization  
- Survey profile mapping
- Human body detection with hyperbolic pattern analysis
- Amplitude vs depth analysis

Author: Generated for GPR drone integration
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import ndimage, signal
from scipy.interpolate import griddata
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class GPRBScanVisualizer:
    """Comprehensive GPR data visualization and analysis"""
    
    def __init__(self, scan_log: List[Dict], gpr_detector):
        self.scan_log = scan_log
        self.gpr_detector = gpr_detector
        self.scan_results = gpr_detector.get_all_results()
        
        # Human body detection parameters
        self.human_body_length_range = (1.4, 2.0)  # meters
        self.human_body_width_range = (0.3, 0.6)   # meters
        self.human_detection_threshold = 0.4
        self.hyperbolic_detection_threshold = 0.3
        
        # Create output directory
        self.output_dir = getattr(gpr_detector, 'output_dir', 'gpr_outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up color schemes
        self.setup_color_schemes()
        
    def setup_color_schemes(self):
        """Setup custom color schemes for different visualizations"""
        # GPR B-scan colormap
        colors_bscan = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000']
        self.cmap_bscan = LinearSegmentedColormap.from_list('gpr_bscan', colors_bscan)
        
        # Human detection colormap
        colors_human = ['#FFFFFF', '#FFFF00', '#FF8000', '#FF0000', '#8B0000']
        self.cmap_human = LinearSegmentedColormap.from_list('human_detect', colors_human)
        
        # Depth analysis colormap
        self.cmap_depth = plt.cm.viridis
        
    def generate_bscan_image(self, save_path: str = None, show_plot: bool = True):
        """Generate traditional B-scan visualization"""
        if not self.scan_results:
            print("[GPR] No scan data available for B-scan")
            return
            
        print(f"[GPR] Generating B-scan from {len(self.scan_results)} scans...")
        
        # Prepare data
        positions = []
        amplitudes = []
        depths = []
        
        for result in self.scan_results:
            distance = np.sqrt(result.x**2 + result.y**2)  # Distance from origin
            positions.append(distance)
            amplitudes.append(result.amplitude_data)
            depths.append(result.depth_profile)
        
        # Convert to numpy arrays
        positions = np.array(positions)
        depth_profile = depths[0] if depths else np.linspace(0, 3, 150)
        
        # Create B-scan matrix
        bscan_matrix = np.array(amplitudes).T
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Main B-scan plot
        extent = [positions.min(), positions.max(), depth_profile.max(), depth_profile.min()]
        im1 = ax1.imshow(bscan_matrix, aspect='auto', extent=extent, 
                        cmap=self.cmap_bscan, interpolation='bilinear')
        
        ax1.set_xlabel('Distance Along Profile (m)')
        ax1.set_ylabel('Depth (m)')
        ax1.set_title('GPR B-Scan Profile - Subsurface Imaging')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Signal Amplitude', rotation=270, labelpad=20)
        
        # Mark detected objects
        for i, result in enumerate(self.scan_results):
            if result.has_detection:
                ax1.axvline(x=positions[i], color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax1.text(positions[i], 0.1, f'DET\n{result.detection_confidence:.2f}', 
                        ha='center', va='top', color='red', fontweight='bold', fontsize=8)
        
        # Amplitude vs position plot
        max_amplitudes = [result.max_amplitude for result in self.scan_results]
        ax2.plot(positions, max_amplitudes, 'b-', linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=self.gpr_detector.threshold, color='r', linestyle='--', 
                   label=f'Detection Threshold ({self.gpr_detector.threshold})')
        ax2.set_xlabel('Distance Along Profile (m)')
        ax2.set_ylabel('Maximum Amplitude')
        ax2.set_title('Detection Amplitude Profile')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Highlight detections
        for i, (pos, amp) in enumerate(zip(positions, max_amplitudes)):
            if self.scan_results[i].has_detection:
                ax2.scatter(pos, amp, color='red', s=100, marker='*', zorder=5)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'gpr_bscan_profile.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        print(f"[GPR] B-scan saved to {save_path}")
        return save_path
    
    def generate_bscan_with_human_detection(self) -> List[Dict]:
        """Generate B-scan with specialized human body detection using hyperbolic analysis"""
        if not self.scan_results:
            print("[GPR] No scan data available for human detection")
            return []
            
        print("[GPR] Analyzing B-scan data for human body signatures...")
        
        # Prepare data for analysis
        positions = []
        amplitudes = []
        detections = []
        
        for result in self.scan_results:
            distance = np.sqrt(result.x**2 + result.y**2)
            positions.append(distance)
            amplitudes.append(result.amplitude_data)
            detections.append(result.has_detection)
        
        positions = np.array(positions)
        bscan_matrix = np.array(amplitudes).T
        depth_profile = self.scan_results[0].depth_profile
        
        # Human body detection algorithm
        human_detections = self._detect_human_bodies(bscan_matrix, positions, depth_profile)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Raw B-scan with human detection overlay
        ax1 = axes[0, 0]
        extent = [positions.min(), positions.max(), depth_profile.max(), depth_profile.min()]
        im1 = ax1.imshow(bscan_matrix, aspect='auto', extent=extent, 
                        cmap=self.cmap_bscan, interpolation='bilinear')
        
        # Overlay human detections
        for detection in human_detections:
            rect = patches.Rectangle(
                (detection['x_start'], detection['depth_start']),
                detection['length_meters'], detection['depth_extent'],
                linewidth=3, edgecolor='yellow', facecolor='none', linestyle='--'
            )
            ax1.add_patch(rect)
            ax1.text(detection['x_center'], detection['depth_center'], 
                    f"HUMAN\n{detection['confidence']:.2f}", 
                    ha='center', va='center', color='yellow', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax1.set_xlabel('Distance Along Profile (m)')
        ax1.set_ylabel('Depth (m)')
        ax1.set_title('GPR B-Scan with Human Body Detection')
        plt.colorbar(im1, ax=ax1, shrink=0.7)
        
        # 2. Hyperbolic pattern analysis
        ax2 = axes[0, 1]
        hyperbolic_analysis = self._analyze_hyperbolic_patterns(bscan_matrix, positions, depth_profile)
        im2 = ax2.imshow(hyperbolic_analysis, aspect='auto', extent=extent, 
                        cmap=self.cmap_human, interpolation='bilinear')
        ax2.set_xlabel('Distance Along Profile (m)')
        ax2.set_ylabel('Depth (m)')
        ax2.set_title('Hyperbolic Pattern Analysis')
        plt.colorbar(im2, ax=ax2, shrink=0.7)
        
        # 3. Human body size analysis
        ax3 = axes[1, 0]
        if human_detections:
            sizes = [det['length_meters'] for det in human_detections]
            widths = [det['width_m'] for det in human_detections]
            confidences = [det['confidence'] for det in human_detections]
            
            scatter = ax3.scatter(sizes, widths, c=confidences, s=200, 
                                cmap='RdYlGn', alpha=0.7, edgecolors='black')
            
            # Add human body size reference box
            human_rect = patches.Rectangle(
                (self.human_body_length_range[0], self.human_body_width_range[0]),
                self.human_body_length_range[1] - self.human_body_length_range[0],
                self.human_body_width_range[1] - self.human_body_width_range[0],
                linewidth=2, edgecolor='red', facecolor='none', linestyle=':'
            )
            ax3.add_patch(human_rect)
            ax3.text(1.7, 0.45, 'Human Body\nSize Range', ha='center', va='center', 
                    color='red', fontweight='bold')
            
            plt.colorbar(scatter, ax=ax3, shrink=0.7, label='Detection Confidence')
        else:
            ax3.text(0.5, 0.5, 'No Human Bodies Detected', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=16, color='gray')
        
        ax3.set_xlabel('Length (m)')
        ax3.set_ylabel('Width (m)')
        ax3.set_title('Human Body Size Analysis')
        ax3.grid(True, alpha=0.3)
        
        # 4. Detection confidence profile
        ax4 = axes[1, 1]
        if human_detections:
            x_centers = [det['x_center'] for det in human_detections]
            confidences = [det['confidence'] for det in human_detections]
            
            ax4.bar(range(len(human_detections)), confidences, 
                   color=['red' if c > 0.7 else 'orange' if c > 0.5 else 'yellow' for c in confidences])
            ax4.axhline(y=self.human_detection_threshold, color='red', linestyle='--', 
                       label=f'Human Detection Threshold ({self.human_detection_threshold})')
            ax4.set_xlabel('Detection Index')
            ax4.set_ylabel('Confidence Score')
            ax4.set_title('Human Detection Confidence Scores')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add labels
            for i, (conf, x_pos) in enumerate(zip(confidences, x_centers)):
                ax4.text(i, conf + 0.02, f'{conf:.2f}\n@{x_pos:.1f}m', 
                        ha='center', va='bottom', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No Human Bodies Detected', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=16, color='gray')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, 'human_body_detection_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"[GPR] Human detection analysis saved to {save_path}")
        
        # Print detection summary
        if human_detections:
            print(f"\n=== HUMAN BODY DETECTION RESULTS ===")
            for i, detection in enumerate(human_detections):
                print(f"Detection {i+1}:")
                print(f"  Position: {detection['x_center']:.1f}m along profile")
                print(f"  Depth: {detection['depth_center']:.1f}m")
                print(f"  Size: {detection['length_meters']:.1f}m × {detection['width_m']:.1f}m")
                print(f"  Confidence: {detection['confidence']:.3f}")
                print(f"  Status: {'HIGH CONFIDENCE' if detection['confidence'] > 0.7 else 'MEDIUM CONFIDENCE' if detection['confidence'] > 0.5 else 'LOW CONFIDENCE'}")
                print()
        
        return human_detections
    
    def _detect_human_bodies(self, bscan_matrix: np.ndarray, positions: np.ndarray, 
                           depth_profile: np.ndarray) -> List[Dict]:
        """Detect human bodies using pattern analysis"""
        detections = []
        
        # Apply Gaussian filter to reduce noise
        filtered_data = ndimage.gaussian_filter(bscan_matrix, sigma=1.0)
        
        # Find connected regions with high amplitude
        threshold = np.mean(filtered_data) + 2 * np.std(filtered_data)
        binary_mask = filtered_data > threshold
        
        # Label connected components
        labeled_array, num_features = ndimage.label(binary_mask)
        
        for i in range(1, num_features + 1):
            # Get region properties
            region_mask = labeled_array == i
            region_coords = np.where(region_mask)
            
            if len(region_coords[0]) < 10:  # Skip small regions
                continue
            
            # Calculate region dimensions
            depth_indices = region_coords[0]
            position_indices = region_coords[1]
            
            depth_start = depth_profile[depth_indices.min()]
            depth_end = depth_profile[depth_indices.max()]
            depth_extent = depth_end - depth_start
            depth_center = (depth_start + depth_end) / 2
            
            x_start = positions[position_indices.min()]
            x_end = positions[position_indices.max()]
            length_meters = x_end - x_start
            x_center = (x_start + x_end) / 2
            
            # Estimate width (this is simplified - in real GPR, width estimation is complex)
            width_m = depth_extent * 0.7  # Approximation based on GPR physics
            
            # Calculate confidence based on size match to human body
            length_match = self._calculate_size_match(length_meters, self.human_body_length_range)
            width_match = self._calculate_size_match(width_m, self.human_body_width_range)
            
            # Calculate signal strength confidence
            region_amplitudes = filtered_data[region_mask]
            signal_confidence = min(np.mean(region_amplitudes) / threshold, 1.0)
            
            # Check for hyperbolic pattern
            hyperbolic_confidence = self._check_hyperbolic_pattern(
                filtered_data, region_coords, positions, depth_profile
            )
            
            # Combined confidence score
            confidence = (length_match * 0.3 + width_match * 0.2 + 
                         signal_confidence * 0.3 + hyperbolic_confidence * 0.2)
            
            # Only consider detections above threshold
            if confidence > self.human_detection_threshold:
                detection = {
                    'x_start': x_start,
                    'x_end': x_end,
                    'x_center': x_center,
                    'depth_start': depth_start,
                    'depth_end': depth_end,
                    'depth_center': depth_center,
                    'depth_extent': depth_extent,
                    'length_meters': length_meters,
                    'width_m': width_m,
                    'confidence': confidence,
                    'signal_strength': signal_confidence,
                    'hyperbolic_score': hyperbolic_confidence
                }
                detections.append(detection)
        
        return detections
    
    def _calculate_size_match(self, measured_size: float, expected_range: Tuple[float, float]) -> float:
        """Calculate how well a measured size matches expected range"""
        min_size, max_size = expected_range
        if min_size <= measured_size <= max_size:
            return 1.0
        elif measured_size < min_size:
            return max(0, 1 - (min_size - measured_size) / min_size)
        else:
            return max(0, 1 - (measured_size - max_size) / max_size)
    
    def _check_hyperbolic_pattern(self, data: np.ndarray, region_coords: Tuple, 
                                positions: np.ndarray, depth_profile: np.ndarray) -> float:
        """Check for hyperbolic reflection pattern characteristic of buried objects"""
        try:
            depth_indices, pos_indices = region_coords
            
            # Extract the region
            min_depth_idx, max_depth_idx = depth_indices.min(), depth_indices.max()
            min_pos_idx, max_pos_idx = pos_indices.min(), pos_indices.max()
            
            if max_depth_idx - min_depth_idx < 5 or max_pos_idx - min_pos_idx < 5:
                return 0.0
            
            region_data = data[min_depth_idx:max_depth_idx+1, min_pos_idx:max_pos_idx+1]
            
            # Look for hyperbolic curvature by analyzing the peak positions at each depth
            peak_positions = []
            for i in range(region_data.shape[0]):
                row = region_data[i, :]
                if row.max() > row.mean() + row.std():
                    peak_pos = np.argmax(row)
                    peak_positions.append(peak_pos)
            
            if len(peak_positions) < 3:
                return 0.0
            
            # Check if peak positions follow a hyperbolic pattern
            # For a hyperbola, the curvature should be convex
            peak_positions = np.array(peak_positions)
            
            # Calculate second derivative to check curvature
            if len(peak_positions) >= 3:
                second_derivative = np.diff(peak_positions, 2)
                curvature_score = np.mean(np.abs(second_derivative))
                
                # Normalize and return score
                return min(curvature_score / 2.0, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_hyperbolic_patterns(self, bscan_matrix: np.ndarray, positions: np.ndarray, 
                                   depth_profile: np.ndarray) -> np.ndarray:
        """Analyze hyperbolic reflection patterns in the data"""
        # Apply edge detection to find hyperbolic patterns
        sobel_h = ndimage.sobel(bscan_matrix, axis=0)  # Horizontal edges
        sobel_v = ndimage.sobel(bscan_matrix, axis=1)  # Vertical edges
        
        # Combine edge detections
        edges = np.sqrt(sobel_h**2 + sobel_v**2)
        
        # Apply Gaussian filter to smooth
        hyperbolic_map = ndimage.gaussian_filter(edges, sigma=1.5)
        
        # Normalize
        hyperbolic_map = (hyperbolic_map - hyperbolic_map.min()) / (hyperbolic_map.max() - hyperbolic_map.min())
        
        return hyperbolic_map
    
    def create_3d_visualization(self, save_path: str = None):
        """Create 3D visualization of GPR survey data"""
        if not self.scan_results:
            print("[GPR] No scan data available for 3D visualization")
            return
            
        print("[GPR] Generating 3D GPR visualization...")
        
        # Prepare data
        x_coords = [result.x for result in self.scan_results]
        y_coords = [result.y for result in self.scan_results]
        z_coords = [result.z for result in self.scan_results]
        amplitudes = [result.max_amplitude for result in self.scan_results]
        detections = [result.has_detection for result in self.scan_results]
        
        # Create 3D plot
        fig = plt.figure(figsize=(16, 12))
        
        # Main 3D scatter plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Color code by detection status
        colors = ['red' if det else 'blue' for det in detections]
        sizes = [50 + 200 * amp for amp in amplitudes]
        
        scatter = ax1.scatter(x_coords, y_coords, z_coords, c=colors, s=sizes, alpha=0.6)
        
        # Add buried objects
        for obj in self.gpr_detector.simulator.buried_objects:
            ax1.scatter(obj.x, obj.y, -obj.depth, c='green', s=300, marker='^', 
                       label='Buried Object' if obj == self.gpr_detector.simulator.buried_objects[0] else "")
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_zlabel('Z Position (m)')
        ax1.set_title('3D GPR Survey - Scan Positions and Detections')
        
        # 2D projection - XY plane
        ax2 = fig.add_subplot(222)
        scatter2 = ax2.scatter(x_coords, y_coords, c=amplitudes, s=sizes, 
                              cmap='viridis', alpha=0.7)
        
        # Add buried objects
        for obj in self.gpr_detector.simulator.buried_objects:
            circle = plt.Circle((obj.x, obj.y), obj.size/2, fill=False, 
                               edgecolor='red', linewidth=2)
            ax2.add_patch(circle)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Survey Area - Top View')
        plt.colorbar(scatter2, ax=ax2, label='Max Amplitude')
        ax2.set_aspect('equal')
        
        # Depth profile
        ax3 = fig.add_subplot(223)
        if self.scan_results:
            # Create interpolated depth map
            xi = np.linspace(min(x_coords), max(x_coords), 50)
            yi = np.linspace(min(y_coords), max(y_coords), 50)
            XI, YI = np.meshgrid(xi, yi)
            
            # Interpolate amplitudes
            ZI = griddata((x_coords, y_coords), amplitudes, (XI, YI), method='cubic', fill_value=0)
            
            contour = ax3.contourf(XI, YI, ZI, levels=20, cmap='viridis')
            plt.colorbar(contour, ax=ax3, label='Signal Amplitude')
            
            # Mark detection points
            for i, (x, y, det) in enumerate(zip(x_coords, y_coords, detections)):
                if det:
                    ax3.scatter(x, y, c='red', s=100, marker='*', edgecolors='white')
        
        ax3.set_xlabel('X Position (m)')
        ax3.set_ylabel('Y Position (m)')
        ax3.set_title('Interpolated Amplitude Map')
        ax3.set_aspect('equal')
        
        # Statistics panel
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        # Calculate statistics
        total_scans = len(self.scan_results)
        total_detections = sum(detections)
        detection_rate = total_detections / total_scans if total_scans > 0 else 0
        max_amplitude = max(amplitudes) if amplitudes else 0
        avg_amplitude = np.mean(amplitudes) if amplitudes else 0
        
        survey_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords)) if x_coords else 0
        
        stats_text = f"""
        === GPR SURVEY STATISTICS ===
        
        Survey Area: {survey_area:.1f} m²
        Total Scan Points: {total_scans}
        Detections Found: {total_detections}
        Detection Rate: {detection_rate:.1%}
        
        Signal Analysis:
        Max Amplitude: {max_amplitude:.4f}
        Avg Amplitude: {avg_amplitude:.4f}
        
        Buried Objects in Area: {len(self.gpr_detector.simulator.buried_objects)}
        
        Survey Efficiency:
        Points per m²: {total_scans/survey_area if survey_area > 0 else 0:.2f}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'gpr_3d_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"[GPR] 3D visualization saved to {save_path}")
        return save_path
    
    def generate_survey_profile_map(self, save_path: str = None):
        """Generate comprehensive survey profile map showing all buried objects"""
        if not self.scan_results:
            print("[GPR] No scan data available for survey map")
            return
            
        print("[GPR] Generating survey profile map...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Detection probability map
        ax1 = axes[0, 0]
        x_coords = [result.x for result in self.scan_results]
        y_coords = [result.y for result in self.scan_results]
        confidences = [result.detection_confidence for result in self.scan_results]
        
        # Create interpolated confidence map
        xi = np.linspace(min(x_coords), max(x_coords), 100)
        yi = np.linspace(min(y_coords), max(y_coords), 100)
        XI, YI = np.meshgrid(xi, yi)
        
        ZI = griddata((x_coords, y_coords), confidences, (XI, YI), method='cubic', fill_value=0)
        
        contour1 = ax1.contourf(XI, YI, ZI, levels=20, cmap='RdYlBu_r')
        plt.colorbar(contour1, ax=ax1, label='Detection Confidence')
        
        # Mark actual buried objects
        for obj in self.gpr_detector.simulator.buried_objects:
            circle = plt.Circle((obj.x, obj.y), obj.size/2, fill=False, 
                               edgecolor='black', linewidth=3, linestyle='--')
            ax1.add_patch(circle)
            ax1.text(obj.x, obj.y, obj.material_type[:4].upper(), ha='center', va='center',
                    fontweight='bold', color='black', fontsize=8)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Detection Confidence Map')
        ax1.set_aspect('equal')
        
        # 2. Material type distribution
        ax2 = axes[0, 1]
        material_counts = {}
        for obj in self.gpr_detector.simulator.buried_objects:
            material_counts[obj.material_type] = material_counts.get(obj.material_type, 0) + 1
        
        if material_counts:
            materials = list(material_counts.keys())
            counts = list(material_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(materials)))
            
            wedges, texts, autotexts = ax2.pie(counts, labels=materials, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            ax2.set_title('Buried Object Material Distribution')
        
        # 3. Depth analysis
        ax3 = axes[1, 0]
        depths = [obj.depth for obj in self.gpr_detector.simulator.buried_objects]
        materials = [obj.material_type for obj in self.gpr_detector.simulator.buried_objects]
        
        # Create scatter plot of depth vs material
        material_types = list(set(materials))
        colors = plt.cm.tab10(np.linspace(0, 1, len(material_types)))
        
        for i, mat_type in enumerate(material_types):
            mat_depths = [d for d, m in zip(depths, materials) if m == mat_type]
            ax3.scatter([i] * len(mat_depths), mat_depths, c=[colors[i]], 
                       s=100, alpha=0.7, label=mat_type)
        
        ax3.set_xlabel('Material Type Index')
        ax3.set_ylabel('Depth (m)')
        ax3.set_title('Object Depth by Material Type')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()  # Deeper objects at bottom
        
        # 4. Survey coverage analysis
        ax4 = axes[1, 1]
        
        # Calculate survey grid coverage
        scan_positions = [(result.x, result.y) for result in self.scan_results]
        x_scan, y_scan = zip(*scan_positions) if scan_positions else ([], [])
        
        # Show scan pattern
        ax4.scatter(x_scan, y_scan, c='blue', s=20, alpha=0.6, label='Scan Points')
        
        # Show buried objects
        for obj in self.gpr_detector.simulator.buried_objects:
            circle = plt.Circle((obj.x, obj.y), obj.size/2, fill=True, 
                               facecolor='red', alpha=0.3, edgecolor='red', linewidth=2)
            ax4.add_patch(circle)
        
        # Calculate coverage statistics
        if scan_positions:
            x_range = max(x_scan) - min(x_scan)
            y_range = max(y_scan) - min(y_scan)
            scan_density = len(scan_positions) / (x_range * y_range) if x_range > 0 and y_range > 0 else 0
            
            ax4.text(0.02, 0.98, f'Scan Density: {scan_density:.2f} points/m²', 
                    transform=ax4.transAxes, va='top', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax4.set_xlabel('X Position (m)')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title('Survey Coverage Pattern')
        ax4.legend()
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'gpr_survey_profile_map.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"[GPR] Survey profile map saved to {save_path}")
        return save_path
    
    def generate_amplitude_depth_analysis(self, save_path: str = None):
        """Generate detailed amplitude vs depth analysis"""
        if not self.scan_results:
            print("[GPR] No scan data available for amplitude analysis")
            return
            
        print("[GPR] Generating amplitude vs depth analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Amplitude vs depth profile
        ax1 = axes[0, 0]
        
        # Combine all amplitude data
        all_amplitudes = []
        all_depths = []
        
        for result in self.scan_results:
            depths = result.depth_profile
            amplitudes = result.amplitude_data
            all_depths.extend(depths)
            all_amplitudes.extend(amplitudes)
        
        # Create scatter plot
        ax1.scatter(all_depths, all_amplitudes, alpha=0.5, s=1)
        
        # Calculate and plot trend line
        if all_depths and all_amplitudes:
            z = np.polyfit(all_depths, all_amplitudes, 2)  # Quadratic fit
            p = np.poly1d(z)
            depth_range = np.linspace(min(all_depths), max(all_depths), 100)
            ax1.plot(depth_range, p(depth_range), "r--", alpha=0.8, linewidth=2, 
                    label='Trend Line')
        
        ax1.set_xlabel('Depth (m)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Amplitude vs Depth Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Depth histogram
        ax2 = axes[0, 1]
        
        # Get depths of detected objects
        detection_depths = []
        for result in self.scan_results:
            if result.has_detection:
                # Find depth of maximum amplitude
                max_idx = np.argmax(result.amplitude_data)
                detection_depths.append(result.depth_profile[max_idx])
        
        if detection_depths:
            ax2.hist(detection_depths, bins=20, alpha=0.7, color='red', edgecolor='black')
            ax2.axvline(np.mean(detection_depths), color='blue', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(detection_depths):.2f}m')
        
        ax2.set_xlabel('Depth (m)')
        ax2.set_ylabel('Number of Detections')
        ax2.set_title('Detection Depth Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Signal attenuation analysis
        ax3 = axes[1, 0]
        
        # Calculate signal attenuation by depth
        depth_bins = np.linspace(0, max(all_depths) if all_depths else 3, 20)
        bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
        mean_amplitudes = []
        
        for i in range(len(depth_bins)-1):
            mask = (np.array(all_depths) >= depth_bins[i]) & (np.array(all_depths) < depth_bins[i+1])
            if np.any(mask):
                mean_amplitudes.append(np.mean(np.array(all_amplitudes)[mask]))
            else:
                mean_amplitudes.append(0)
        
        ax3.plot(bin_centers, mean_amplitudes, 'b-o', linewidth=2, markersize=6)
        ax3.set_xlabel('Depth (m)')
        ax3.set_ylabel('Mean Amplitude')
        ax3.set_title('Signal Attenuation with Depth')
        ax3.grid(True, alpha=0.3)
        
        # Fit exponential decay
        if len(bin_centers) > 3 and max(mean_amplitudes) > 0:
            try:
                # Fit exponential decay: A * exp(-alpha * depth)
                from scipy.optimize import curve_fit
                
                def exp_decay(x, a, alpha):
                    return a * np.exp(-alpha * x)
                
                popt, _ = curve_fit(exp_decay, bin_centers, mean_amplitudes, 
                                  p0=[max(mean_amplitudes), 0.5])
                
                depth_fit = np.linspace(0, max(bin_centers), 100)
                ax3.plot(depth_fit, exp_decay(depth_fit, *popt), 'r--', 
                        label=f'Exp fit: α={popt[1]:.2f}')
                ax3.legend()
            except:
                pass
        
        # 4. Detection efficiency by depth
        ax4 = axes[1, 1]
        
        # Calculate detection rate by depth
        detection_rates = []
        depth_centers = []
        
        for i in range(len(depth_bins)-1):
            depth_mask = (np.array(all_depths) >= depth_bins[i]) & (np.array(all_depths) < depth_bins[i+1])
            
            if np.any(depth_mask):
                # Count detections in this depth range
                total_points = np.sum(depth_mask)
                detections_in_range = 0
                
                for result in self.scan_results:
                    for j, depth in enumerate(result.depth_profile):
                        if depth_bins[i] <= depth < depth_bins[i+1]:
                            if result.amplitude_data[j] > self.gpr_detector.threshold:
                                detections_in_range += 1
                                break
                
                detection_rate = detections_in_range / len(self.scan_results) if len(self.scan_results) > 0 else 0
                detection_rates.append(detection_rate)
                depth_centers.append((depth_bins[i] + depth_bins[i+1]) / 2)
        
        if detection_rates:
            ax4.bar(depth_centers, detection_rates, width=np.diff(depth_bins)[0]*0.8, 
                   alpha=0.7, color='green', edgecolor='black')
        
        ax4.set_xlabel('Depth (m)')
        ax4.set_ylabel('Detection Rate')
        ax4.set_title('Detection Efficiency by Depth')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'amplitude_depth_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"[GPR] Amplitude-depth analysis saved to {save_path}")
        return save_path
    
    def generate_comprehensive_report(self, save_path: str = None):
        """Generate comprehensive GPR analysis report"""
        if not self.scan_results:
            print("[GPR] No scan data available for report")
            return
            
        print("[GPR] Generating comprehensive GPR report...")
        
        # Collect all analysis data
        total_scans = len(self.scan_results)
        total_detections = sum(1 for result in self.scan_results if result.has_detection)
        detection_rate = total_detections / total_scans if total_scans > 0 else 0
        
        # Calculate survey area
        x_coords = [result.x for result in self.scan_results]
        y_coords = [result.y for result in self.scan_results]
        
        if x_coords and y_coords:
            survey_area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
            scan_density = total_scans / survey_area if survey_area > 0 else 0
        else:
            survey_area = 0
            scan_density = 0
        
        # Signal statistics
        max_amplitudes = [result.max_amplitude for result in self.scan_results]
        max_amplitude = max(max_amplitudes) if max_amplitudes else 0
        avg_amplitude = np.mean(max_amplitudes) if max_amplitudes else 0
        std_amplitude = np.std(max_amplitudes) if max_amplitudes else 0
        
        # Human body detection analysis
        human_detections = self.generate_bscan_with_human_detection()
        
        # Generate all visualizations
        bscan_path = self.generate_bscan_image(show_plot=False)
        viz_3d_path = self.create_3d_visualization()
        profile_path = self.generate_survey_profile_map()
        
        # Generate all visualizations
        try:
            bscan_path = self.generate_bscan_image(show_plot=False)
        except Exception as e:
            print(f"[GPR] Warning: B-scan generation failed: {e}")
            bscan_path = None
        
        try:
            viz_3d_path = self.create_3d_visualization()
        except Exception as e:
            print(f"[GPR] Warning: 3D visualization failed: {e}")
            viz_3d_path = None
        
        try:
            profile_path = self.generate_survey_profile_map()
        except Exception as e:
            print(f"[GPR] Warning: Profile map failed: {e}")
            profile_path = None
        
        try:
            amplitude_path = self.generate_amplitude_depth_analysis()
        except Exception as e:
            print(f"[GPR] Warning: Amplitude analysis failed: {e}")
            amplitude_path = None
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'survey_summary': {
                'total_scan_points': total_scans,
                'survey_area_m2': round(survey_area, 2),
                'scan_density_per_m2': round(scan_density, 2),
                'total_detections': total_detections,
                'detection_rate_percent': round(detection_rate * 100, 1)
            },
            'signal_analysis': {
                'max_amplitude': round(max_amplitude, 4),
                'average_amplitude': round(avg_amplitude, 4),
                'amplitude_std': round(std_amplitude, 4),
                'detection_threshold': self.gpr_detector.threshold
            },
            'buried_objects': [
                {
                    'material_type': obj.material_type,
                    'position': {'x': obj.x, 'y': obj.y, 'depth': obj.depth},
                    'size': obj.size,
                    'dielectric_constant': getattr(obj, 'dielectric_constant', 2.5)
                }
                for obj in self.gpr_detector.simulator.buried_objects
            ],
            'human_body_detections': [
                {
                    'position_m': round(det['x_center'], 2),
                    'depth_m': round(det['depth_center'], 2),
                    'size_length_m': round(det['length_meters'], 2),
                    'size_width_m': round(det['width_m'], 2),
                    'confidence': round(det['confidence'], 3),
                    'classification': ('HIGH' if det['confidence'] > 0.7 
                                     else 'MEDIUM' if det['confidence'] > 0.5 
                                     else 'LOW')
                }
                for det in human_detections
            ],
            'generated_files': {
                'bscan_profile': bscan_path if bscan_path else 'Failed to generate',
                'human_detection_analysis': 'human_body_detection_analysis.png',
                'visualization_3d': viz_3d_path if viz_3d_path else 'Failed to generate',
                'survey_profile_map': profile_path if profile_path else 'Failed to generate',
                'amplitude_depth_analysis': amplitude_path if amplitude_path else 'Failed to generate'
            }
        }
        
        # Save JSON report
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'gpr_comprehensive_report.json')
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary to console
        print("\n" + "="*60)
        print("GPR SURVEY COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        print(f"Survey Date/Time: {report['timestamp']}")
        print(f"Survey Area: {report['survey_summary']['survey_area_m2']} m²")
        print(f"Total Scan Points: {report['survey_summary']['total_scan_points']}")
        print(f"Scan Density: {report['survey_summary']['scan_density_per_m2']:.2f} points/m²")
        print(f"Detection Rate: {report['survey_summary']['detection_rate_percent']}%")
        print(f"\nSignal Analysis:")
        print(f"  Max Amplitude: {report['signal_analysis']['max_amplitude']}")
        print(f"  Avg Amplitude: {report['signal_analysis']['average_amplitude']}")
        print(f"  Detection Threshold: {report['signal_analysis']['detection_threshold']}")
        
        if report['human_body_detections']:
            print(f"\nHuman Body Detections: {len(report['human_body_detections'])}")
            for i, det in enumerate(report['human_body_detections']):
                print(f"  Detection {i+1}: {det['classification']} confidence")
                print(f"    Position: {det['position_m']}m, Depth: {det['depth_m']}m")
                print(f"    Size: {det['size_length_m']}m × {det['size_width_m']}m")
        else:
            print(f"\nHuman Body Detections: None detected")
        
        print(f"\nBuried Objects in Survey Area: {len(report['buried_objects'])}")
        for obj in report['buried_objects']:
            print(f"  {obj['material_type']} at ({obj['position']['x']:.1f}, {obj['position']['y']:.1f}) "
                  f"depth {obj['position']['depth']:.1f}m")
        
        print(f"\nGenerated Files:")
        for file_type, file_path in report['generated_files'].items():
            print(f"  {file_type}: {file_path}")
        
        print(f"\nFull report saved to: {save_path}")
        print("="*60)
        
        return save_path, report

# Usage example and utility functions
def run_complete_gpr_analysis(gpr_detector, scan_log):
    """Run complete GPR analysis pipeline"""
    print("[GPR] Starting comprehensive GPR analysis...")
    
    # Initialize visualizer
    visualizer = GPRBScanVisualizer(scan_log, gpr_detector)
    
    # Generate all analyses
    try:
        # Generate comprehensive report (includes all other analyses)
        report_path, report_data = visualizer.generate_comprehensive_report()
        print(f"[GPR] Complete analysis finished. Report saved to: {report_path}")
        return visualizer, report_data
        
    except Exception as e:
        print(f"[GPR] Error during analysis: {str(e)}")
        return None, None

# Additional utility functions for GPR analysis
def analyze_detection_patterns(scan_results):
    """Analyze patterns in GPR detections"""
    if not scan_results:
        return {}
    
    detections = [r for r in scan_results if r.has_detection]
    
    if not detections:
        return {'pattern': 'no_detections'}
    
    # Spatial clustering analysis
    positions = [(r.x, r.y) for r in detections]
    
    # Simple clustering - find detection hotspots
    from collections import defaultdict
    grid_size = 0.5  # 0.5m grid cells
    grid_counts = defaultdict(int)
    
    for x, y in positions:
        grid_x = int(x / grid_size)
        grid_y = int(y / grid_size)
        grid_counts[(grid_x, grid_y)] += 1
    
    max_cluster_size = max(grid_counts.values()) if grid_counts else 0
    
    return {
        'total_detections': len(detections),
        'detection_clusters': len(grid_counts),
        'max_cluster_size': max_cluster_size,
        'clustering_pattern': 'clustered' if max_cluster_size > 2 else 'scattered'
    }

def calculate_survey_efficiency(scan_results, buried_objects):
    """Calculate survey efficiency metrics"""
    if not scan_results or not buried_objects:
        return {}
    
    # Calculate detection success rate for known objects
    detected_objects = 0
    
    for obj in buried_objects:
        # Check if any scan near this object detected something
        for result in scan_results:
            distance = np.sqrt((result.x - obj.x)**2 + (result.y - obj.y)**2)
            if distance < 1.0 and result.has_detection:  # Within 1m
                detected_objects += 1
                break
    
    success_rate = detected_objects / len(buried_objects) if buried_objects else 0
    
    return {
        'total_buried_objects': len(buried_objects),
        'detected_objects': detected_objects,
        'success_rate': success_rate,
        'false_positive_rate': max(0, len([r for r in scan_results if r.has_detection]) - detected_objects) / len(scan_results)
    }

print("[GPR] GPR B-Scan Visualization and Analysis System - Complete")
print("[GPR] Ready for comprehensive GPR data analysis")