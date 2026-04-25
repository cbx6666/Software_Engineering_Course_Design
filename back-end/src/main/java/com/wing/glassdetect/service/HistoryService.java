package com.wing.glassdetect.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.wing.glassdetect.mapper.HistoryMapper;
import com.wing.glassdetect.model.History;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Collections;
import java.util.List;

@Service
public class HistoryService {

    private final HistoryMapper historyMapper;
    private final boolean persistenceEnabled;

    public HistoryService(HistoryMapper historyMapper,
                          @Value("${app.persistence.enabled:true}") boolean persistenceEnabled) {
        this.historyMapper = historyMapper;
        this.persistenceEnabled = persistenceEnabled;
    }

    /**
     * 根据用户ID获取历史记录列表
     * @param userId 用户ID
     * @return 历史记录列表
     */
    public List<History> getHistoryByUserId(Long userId) {
        if (!persistenceEnabled) {
            return Collections.emptyList();
        }
        LambdaQueryWrapper<History> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(History::getUserId, userId)
                    .orderByDesc(History::getDate); // 按日期降序排序
        return historyMapper.selectList(queryWrapper);
    }

    /**
     * 根据ID获取单个历史记录
     * @param id 历史记录ID
     * @return 单个历史记录
     */
    public History getHistoryById(Long id) {
        if (!persistenceEnabled) {
            return null;
        }
        return historyMapper.selectById(id);
    }

    /**
     * 保存一条历史记录
     * @param history 历史记录对象
     */
    @Transactional
    public void saveHistory(History history) {
        if (!persistenceEnabled) {
            return;
        }
        System.out.println("Saving original images: " + history.getOriginalImages());
        historyMapper.insert(history);
    }
}
