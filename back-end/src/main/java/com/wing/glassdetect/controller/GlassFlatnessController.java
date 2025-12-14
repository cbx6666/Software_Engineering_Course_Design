package com.wing.glassdetect.controller;

import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.service.GlassFlatnessService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/api/detect/glass-flatness")
public class GlassFlatnessController {

    @Autowired
    private GlassFlatnessService glassFlatnessService;

    @Value("${algorithm.url}")
    private String algorithmUrl;

    @PostMapping
    public CompletableFuture<DetectionResult> detectGlassFlatness(@RequestParam("left_env") MultipartFile leftEnv,
                                                                  @RequestParam("left_mix") MultipartFile leftMix,
                                                                  @RequestParam("right_env") MultipartFile rightEnv,
                                                                  @RequestParam("right_mix") MultipartFile rightMix) {

        MultipartFile[] images = new MultipartFile[]{leftEnv, leftMix, rightEnv, rightMix};
        String[] fieldNames = {"left_env", "left_mix", "right_env", "right_mix"};
        String url = algorithmUrl + "/api/detect/glass-flatness";

        return glassFlatnessService.detect(images, fieldNames, url);
    }
}