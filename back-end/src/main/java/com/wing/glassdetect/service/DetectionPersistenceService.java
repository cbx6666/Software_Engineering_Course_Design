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
    private final boolean persistenceEnabled;

    @Autowired
    public DetectionPersistenceService(HistoryService historyService,
                                       ObjectMapper objectMapper,
                                       @Value("${image.storage.path}") String imageStoragePath,
                                       @Value("${app.persistence.enabled:true}") boolean persistenceEnabled) {
        this.historyService = historyService;
        this.objectMapper = objectMapper;
        this.imageStoragePath = imageStoragePath;
        this.persistenceEnabled = persistenceEnabled;
    }

    public void persistResult(String email, String type, DetectionResult result, Path[] tempOriginalFiles) throws IOException {
        String webImagePath = normalizeResultImagePath(result);

        if (!persistenceEnabled) {
            System.out.println("Persistence disabled; skip saving detection result for local development.");
            return;
        }

        History history = new History();
        history.setEmail(email);
        history.setType(type);
        history.setDate(LocalDateTime.now());
        history.setStatus(result.getStatus());
        history.setTitle(result.getTitle());
        history.setDescription(result.getDescription());

        String taskFolderName = UUID.randomUUID().toString();
        Path taskStoragePath = Paths.get(imageStoragePath, taskFolderName);
        Files.createDirectories(taskStoragePath);

        if (tempOriginalFiles != null && tempOriginalFiles.length > 0) {
            List<String> originalImagePaths = saveOriginalImages(tempOriginalFiles, taskStoragePath, taskFolderName);
            history.setOriginalImages(originalImagePaths);
        }

        if ("flatness".equals(type) && result.getPointcloud() != null) {
            Map<String, Object> pointcloudMap = objectMapper.convertValue(result.getPointcloud(), new TypeReference<>() {});
            history.setPointcloud(pointcloudMap);
        }

        if (result.getDetails() != null) {
            List<Map<String, String>> detailsList = objectMapper.convertValue(result.getDetails(), new TypeReference<>() {});
            history.setDetails(detailsList);
        }

        if (webImagePath != null) {
            history.setImage(webImagePath);
        }

        historyService.saveHistory(history);
    }

    private String normalizeResultImagePath(DetectionResult result) {
        if (result == null || result.getImage() == null || result.getImage().isEmpty()) {
            return null;
        }

        String image = result.getImage().replace('\\', '/');
        int resultRootIndex = image.toLowerCase().indexOf("/data/result/");
        if (resultRootIndex >= 0) {
            String webImagePath = "/results/" + image.substring(resultRootIndex + "/data/result/".length());
            result.setImage(webImagePath);
            return webImagePath;
        }

        if (image.startsWith("results/")) {
            image = "/" + image;
        }

        if (image.startsWith("/results/")) {
            result.setImage(image);
            return image;
        }

        return image;
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

            savedImagePaths.add("/images/" + taskFolderName + "/" + newFileName);
        }
        return savedImagePaths;
    }
}
