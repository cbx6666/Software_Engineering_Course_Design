package com.wing.glassdetect.service;

import com.wing.glassdetect.model.DetectionResult;
import com.wing.glassdetect.utils.ApiUtils;
import com.wing.glassdetect.utils.FileUtils;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Path;

@Service
public class GlassCrackService {

    public DetectionResult detect(MultipartFile[] imageFiles, String url) {
        Path[] tempFiles = null;
        try {
            tempFiles = FileUtils.saveTempFile(imageFiles);
            return ApiUtils.postImage(tempFiles, url);
        } catch (IOException e) {
            return new DetectionResult("error", "上传图片失败", e.getMessage(), null);
        } finally {
            if (tempFiles != null) {
                for (Path p : tempFiles) {
                    FileUtils.deleteTempFile(p);
                }
            }
        }
    }
}
