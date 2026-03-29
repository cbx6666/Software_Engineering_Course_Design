package com.wing.glassdetect.dto;

import com.wing.glassdetect.model.DetectionResult;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.nio.file.Path;

@Getter
@AllArgsConstructor
public class DetectionTaskResultDto {
    private final DetectionResult detectionResult;
    private final Path tempDirectory;
    private final Path[] tempFiles;
}

