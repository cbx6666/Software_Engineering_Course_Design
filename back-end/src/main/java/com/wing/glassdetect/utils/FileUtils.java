package com.wing.glassdetect.utils;

import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.*;
import java.util.UUID;


public class FileUtils {
    public static Path saveTempFile(MultipartFile imageFile) throws IOException {
        Path tempDir = Paths.get(System.getProperty("user.dir"), "imageUploaded");
        if (!Files.exists(tempDir)) {
            Files.createDirectories(tempDir);
        }

        Path tempFile = tempDir.resolve("upload-" + System.currentTimeMillis() + "-" + UUID.randomUUID() + ".jpg");
        Files.copy(imageFile.getInputStream(), tempFile, StandardCopyOption.REPLACE_EXISTING);

        return tempFile;
    }

    public static void deleteTempFile(Path tempFile) {
        if (tempFile != null) {
            try {
                Files.deleteIfExists(tempFile);
            } catch (IOException ignored) {}
        }
    }
}
