package com.wing.glassdetect.controller;

import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.service.GlassCrackService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/api/detect/glass-crack")
public class GlassCrackController {

    @Autowired
    private GlassCrackService glassCrackService;

    @Value("${algorithm.url}")
    private String algorithmUrl;

    @PostMapping
    public CompletableFuture<DetectionResult> detectGlassCrack(@RequestParam("images") MultipartFile[] images) {
        String url =  algorithmUrl + "/api/detect/glass-crack";

        return glassCrackService.detect(images, url);
    }
}