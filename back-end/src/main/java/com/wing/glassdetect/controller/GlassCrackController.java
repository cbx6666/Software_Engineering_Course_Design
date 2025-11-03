package com.wing.glassdetect.controller;

import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.service.GlassCrackService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/detect/glass-crack")
@CrossOrigin(origins = "*") // 允许前端跨域访问
public class GlassCrackController {

    @Autowired
    private GlassCrackService glassCrackService;

    @PostMapping
    public DetectionResult detectGlassCrack(@RequestParam("image") MultipartFile image) {
        return glassCrackService.detect(image);
    }
}