package com.wing.glassdetect.dto.auth;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 返回结构（不再返回 token）：
 * { user: { id: string, email: string } }
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class LoginResponse {
    private User user;

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class User {
        private String id;
        private String email;
    }
}
