package com.wing.glassdetect.controller;

import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.service.GlassFlatnessService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/detect/glass-flatness")
@CrossOrigin(origins = "*") // 允许前端跨域访问
public class GlassFlatnessController {

    @Autowired
    private GlassFlatnessService glassFlatnessService;

    @PostMapping
    public DetectionResult detectGlassFlatness(@RequestParam("image") MultipartFile image) {
        return glassFlatnessService.detect(image);
    }
}