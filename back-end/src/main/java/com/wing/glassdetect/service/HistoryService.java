package com.wing.glassdetect.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.wing.glassdetect.mapper.HistoryMapper;
import com.wing.glassdetect.model.History;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class HistoryService {

    private final HistoryMapper historyMapper;

    public HistoryService(HistoryMapper historyMapper) {
        this.historyMapper = historyMapper;
    }

    /**
     * 根据用户ID获取历史记录列表
     * @param userId 用户ID
     * @return 历史记录列表
     */
    public List<History> getHistoryByUserId(Long userId) {
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
        return historyMapper.selectById(id);
    }

    /**
     * 保存一条历史记录
     * @param history 历史记录对象
     */
    @Transactional
    public void saveHistory(History history) {
        historyMapper.insert(history);
    }
}
