package com.wing.glassdetect.service;

import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.utils.ApiUtils;
import com.wing.glassdetect.utils.FileUtils;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Path;

@Service
public class GlassFlatnessService {

    public DetectionResult detect(MultipartFile imageFile, String url) {
        Path tempFile = null;
        try {
            tempFile = FileUtils.saveTempFile(imageFile);
            return ApiUtils.postImage(tempFile, url);
        } catch (IOException e) {
            return new DetectionResult("error", "上传图片失败", e.getMessage(), null);
        } finally {
            FileUtils.deleteTempFile(tempFile);
        }
    }
}
