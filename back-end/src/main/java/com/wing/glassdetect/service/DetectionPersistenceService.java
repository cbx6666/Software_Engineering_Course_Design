package com.wing.glassdetect.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.model.History;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class DetectionPersistenceService {

    private final HistoryService historyService;
    private final ObjectMapper objectMapper;
    private final String imageStoragePath;

    @Autowired
    public DetectionPersistenceService(HistoryService historyService, 
                                       ObjectMapper objectMapper, 
                                       @Value("${image.storage.path}") String imageStoragePath) {
        this.historyService = historyService;
        this.objectMapper = objectMapper;
        this.imageStoragePath = imageStoragePath;
    }

    public void persistResult(Long userId, String type, DetectionResult result, Path[] tempOriginalFiles) throws IOException {
        History history = new History();
        history.setUserId(userId);
        history.setType(type);
        history.setDate(LocalDateTime.now());
        history.setStatus(result.getStatus());
        history.setTitle(result.getTitle());
        history.setDescription(result.getDescription());

        // 为本次检测创建一个唯一的子目录
        String taskFolderName = UUID.randomUUID().toString();
        Path taskStoragePath = Paths.get(imageStoragePath, taskFolderName);
        Files.createDirectories(taskStoragePath);

        // 1. 保存上传的原图到子目录
        if (tempOriginalFiles != null && tempOriginalFiles.length > 0) {
            List<String> originalImagePaths = saveOriginalImages(tempOriginalFiles, taskStoragePath, taskFolderName);
            history.setOriginalImages(originalImagePaths);
        }

        // 2. 根据类型处理不同的数据
        if ("flatness".equals(type) && result.getPointcloud() != null) {
            Map<String, Object> pointcloudMap = objectMapper.convertValue(result.getPointcloud(), new TypeReference<>() {});
            history.setPointcloud(pointcloudMap);
        }

        if (result.getDetails() != null) {
            List<Map<String, String>> detailsList = objectMapper.convertValue(result.getDetails(), new TypeReference<>() {});
            history.setDetails(detailsList);
        }

        // 3. 保存算法生成的结果图到子目录
        if (result.getImage() != null && !result.getImage().isEmpty()) {
            String webImagePath = result.getImage().replace("/data/result", "/results");
            history.setImage(webImagePath);
            result.setImage(webImagePath); // 更新返回给前端的路径
        }

        historyService.saveHistory(history);
    }

    private List<String> saveOriginalImages(Path[] tempFiles, Path taskStoragePath, String taskFolderName) throws IOException {
        List<String> savedImagePaths = new ArrayList<>();

        for (Path tempFile : tempFiles) {
            String originalFileName = tempFile.getFileName().toString();
            String fileExtension = "";
            int lastDot = originalFileName.lastIndexOf('.');
            if (lastDot > 0) {
                fileExtension = originalFileName.substring(lastDot);
            }

            String newFileName = UUID.randomUUID() + fileExtension;
            Path destinationPath = taskStoragePath.resolve(newFileName);
            Files.copy(tempFile, destinationPath, StandardCopyOption.REPLACE_EXISTING);
            
            // 生成Web可访问路径
            savedImagePaths.add("/images/" + taskFolderName + "/" + newFileName);
        }
        return savedImagePaths;
    }
}
