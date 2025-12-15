package com.wing.glassdetect.model;

import java.util.List;

public class DetectionResult {
    private String status; // success / warning / error
    private String title;
    private String description;
    private List<Detail> details;
    private String image;
    private PointCloudData pointcloud;

    public DetectionResult() {
    }

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

    // Detail definition
    public static class Detail {
        private String label;
        private String value;
        // 新增字段：用于承载前后端/算法返回的指标解释说明
        private String description;

        public Detail() {
        }

        public Detail(String label, String value) {
            this.label = label;
            this.value = value;
        }

        public String getLabel() {
            return label;
        }

        public void setLabel(String label) {
            this.label = label;
        }

        public String getValue() {
            return value;
        }

        public void setValue(String value) {
            this.value = value;
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }
    }

    /**
     * 点云数据结构
     * 
     * 必需字段：
     * - projected_points: 投影后的点坐标 (N×3 array, 单位：米)
     * - projected_dists: 投影后的 Z' 值，用于颜色映射 (N array, 单位：米)
     */
    public static class PointCloudData {
        private List<List<Double>> projected_points; // 投影后的点坐标 (必需)
        private List<Double> projected_dists; // 投影后的 Z' 值 (必需)

        public PointCloudData() {
        }

        public List<List<Double>> getProjected_points() {
            return projected_points;
        }

        public void setProjected_points(List<List<Double>> projected_points) {
            this.projected_points = projected_points;
        }

        public List<Double> getProjected_dists() {
            return projected_dists;
        }

        public void setProjected_dists(List<Double> projected_dists) {
            this.projected_dists = projected_dists;
        }
    }

    /// getter and setter //////////

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public List<Detail> getDetails() {
        return details;
    }

    public void setDetails(List<Detail> details) {
        this.details = details;
    }

    public String getImage() {
        return image;
    }

    public void setImage(String image) {
        this.image = image;
    }

    public PointCloudData getPointcloud() {
        return pointcloud;
    }

    public void setPointcloud(PointCloudData pointcloud) {
        this.pointcloud = pointcloud;
    }
}
