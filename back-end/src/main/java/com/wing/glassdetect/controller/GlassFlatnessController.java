package com.wing.glassdetect.controller;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.model.History;
import com.wing.glassdetect.service.GlassFlatnessService;
import com.wing.glassdetect.service.HistoryService;
import com.wing.glassdetect.utils.FileUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/api/detect/glass-flatness")
public class GlassFlatnessController {

    @Autowired
    private GlassFlatnessService glassFlatnessService;

    @Autowired
    private HistoryService historyService;

    @Autowired
    private ObjectMapper objectMapper;

    @Value("${algorithm.url}")
    private String algorithmUrl;

    @Value("${image.storage.path}")
    private String imageStoragePath;

    @PostMapping
    public ResponseEntity<DetectionResult> detectGlassFlatness(@RequestParam("userId") Long userId,
                                                               @RequestParam("left_env") MultipartFile leftEnv,
                                                               @RequestParam("left_mix") MultipartFile leftMix,
                                                               @RequestParam("right_env") MultipartFile rightEnv,
                                                               @RequestParam("right_mix") MultipartFile rightMix) {

        MultipartFile[] images = new MultipartFile[]{leftEnv, leftMix, rightEnv, rightMix};
        String[] fieldNames = {"left_env", "left_mix", "right_env", "right_mix"};
        String url = algorithmUrl + "/api/detect/glass-flatness";

        CompletableFuture<DetectionResult> future = glassFlatnessService.detect(userId, images, fieldNames, url);
        Path tempImageDir = null;

        try {
            DetectionResult result = future.get(); // 等待异步任务完成

            History history = new History();
            history.setUserId(userId);
            history.setType("flatness");
            history.setDate(LocalDateTime.now());
            history.setStatus(result.getStatus());
            history.setTitle(result.getTitle());
            history.setDescription(result.getDescription());

            if (result.getPointcloud() != null) {
                Map<String, Object> pointcloudMap = objectMapper.convertValue(result.getPointcloud(), new TypeReference<>() {});
                history.setPointcloud(pointcloudMap);
            }
            if (result.getDetails() != null) {
                List<Map<String, String>> detailsList = objectMapper.convertValue(result.getDetails(), new TypeReference<>() {});
                history.setDetails(detailsList);
            }

            // 处理图片存储
            if (result.getImage() != null && !result.getImage().isEmpty()) {
                Path tempImagePath = Paths.get(result.getImage());
                tempImageDir = tempImagePath.getParent();
                String newFileName = UUID.randomUUID().toString() + "_" + tempImagePath.getFileName().toString();
                Path permanentImagePath = Paths.get(imageStoragePath, newFileName);

                Files.createDirectories(permanentImagePath.getParent());
                Files.copy(tempImagePath, permanentImagePath, StandardCopyOption.REPLACE_EXISTING);

                String webImagePath = "/images/" + newFileName;
                history.setImage(webImagePath);
                result.setImage(webImagePath); // 更新返回给前端的路径
            }

            historyService.saveHistory(history);

            return ResponseEntity.ok(result);
        } catch (Exception e) {
            DetectionResult errorResult = new DetectionResult("error", "检测失败", e.getMessage(), null);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResult);
        } finally {
            if (tempImageDir != null) {
                FileUtils.deleteTempDir(tempImageDir);
            }
        }
    }
}
