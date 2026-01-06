import React, { createContext, useContext, useState, ReactNode } from 'react';

interface TabsContextProps {
  activeTab: string;
  setActiveTab: (value: string) => void;
}

const TabsContext = createContext<TabsContextProps | undefined>(undefined);

export function Tabs({ defaultValue, children, className }: { defaultValue: string; children: ReactNode; className?: string }) {
  const [activeTab, setActiveTab] = useState(defaultValue);

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  );
}

export function TabsList({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={`flex border-b border-slate-700 ${className}`}>{children}</div>;
}

export function TabsTrigger({ value, children }: { value: string; children: ReactNode }) {
  const context = useContext(TabsContext);
  if (!context) {
    throw new Error('TabsTrigger must be used within a Tabs component');
  }
  const { activeTab, setActiveTab } = context;
  const isActive = activeTab === value;

  const baseClasses = "px-4 py-2 -mb-px font-semibold border-b-2 transition-colors duration-200 cursor-pointer";
  const activeClasses = "text-cyan-400 border-cyan-400";
  const inactiveClasses = "text-slate-400 border-transparent hover:text-white";

  return (
    <button
      onClick={() => setActiveTab(value)}
      className={`${baseClasses} ${isActive ? activeClasses : inactiveClasses}`}
    >
      {children}
    </button>
  );
}

export function TabsContent({ value, children, className }: { value: string; children: ReactNode; className?: string }) {
  const context = useContext(TabsContext);
  if (!context) {
    throw new Error('TabsContent must be used within a Tabs component');
  }
  const { activeTab } = context;

  return activeTab === value ? <div className={className}>{children}</div> : null;
}

