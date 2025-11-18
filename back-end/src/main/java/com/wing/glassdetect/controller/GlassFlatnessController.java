package com.wing.glassdetect.controller;

import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.service.GlassFlatnessService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/detect/glass-flatness")
public class GlassFlatnessController {

    @Autowired
    private GlassFlatnessService glassFlatnessService;

    @Value("${algorithm.url}")
    private String algorithmUrl;

    @PostMapping
    public DetectionResult detectGlassFlatness(@RequestParam("image") MultipartFile image) {
        String url = algorithmUrl + "/api/detect/glass-flatness";

        return glassFlatnessService.detect(image, url);
    }
}