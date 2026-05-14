package com.wing.glassdetect.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class DetectionResult {
    private String status; // success / warning / error
    private String title;
    private String description;
    private List<Detail> details;
    private String image;
    private PointCloudData pointcloud;

    public DetectionResult(String status, String title, String description, List<Detail> details) {
        this.status = status;
        this.title = title;
        this.description = description;
        this.details = details;
    }

    public DetectionResult(String status, String title, String description, List<Detail> details, String image) {
        this.status = status;
        this.title = title;
        this.description = description;
        this.details = details;
        this.image = image;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Detail {
        private String label;
        private String value;
        private String description;

        public Detail(String label, String value) {
            this.label = label;
            this.value = value;
        }
    }

    @Data
    @NoArgsConstructor
    public static class PointCloudData {
        private List<List<Double>> projected_points;
        private List<Double> projected_dists;
        private FitHeightBand fit_height_band;

        @Data
        @NoArgsConstructor
        public static class FitHeightBand {
            private Boolean enabled;
            private String coordinate_system;
            private String unit;
            private Double lower_z;
            private Double upper_z;
            private List<Double> x_range;
            private List<Double> y_range;
            private List<BoundaryPlane> boundary_planes;
            private RenderHint render_hint;
        }

        @Data
        @NoArgsConstructor
        public static class BoundaryPlane {
            private String name;
            private Double z;
            private List<List<Double>> corners;
        }

        @Data
        @NoArgsConstructor
        public static class RenderHint {
            private String color;
            private Double opacity;
        }
    }
}
