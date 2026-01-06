package com.wing.glassdetect.dto.auth;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class LoginRequest {
    @NotBlank(message = "email 不能为空")
    @Email(message = "email 格式不正确")
    private String email;

    @NotBlank(message = "password 不能为空")
    private String password;
}
