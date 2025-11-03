package com.wing.glassdetect.model;

import java.util.List;

public class DetectionResult {
    private String status; // success / warning / error
    private String title;
    private String description;
    private List<Detail> details;

    public DetectionResult(String status, String title, String description, List<Detail> details) {
        this.status = status;
        this.title = title;
        this.description = description;
        this.details = details;
    }

    // Detail definition
    public static class Detail {
        private String label;
        private String value;

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

}
