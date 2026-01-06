import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { HelpCircle } from "lucide-react";
import type { DetectionDetail } from "@/types/detection";

export function DetailsList({ details }: { details: DetectionDetail[] }) {
    const resultDetails = details.filter((d) => d != null);
    if (resultDetails.length === 0) return null;

    return (
        <div className="space-y-2 bg-slate-900/30 rounded-lg p-4 backdrop-blur-sm">
            <h4 className="text-white text-sm font-medium mb-3">详细指标</h4>
            {resultDetails.map((detail, index) => (
                <div key={index} className="py-2 border-t border-white/10 first:border-t-0">
                    <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                            <span className="text-slate-400">{detail.label}</span>
                            {detail.description && (
                                <Popover>
                                    <PopoverTrigger asChild>
                                        <button className="text-slate-500 hover:text-slate-300 transition-colors focus:outline-none">
                                            <HelpCircle className="w-4 h-4" />
                                        </button>
                                    </PopoverTrigger>
                                    <PopoverContent
                                        side="right"
                                        align="start"
                                        className="bg-slate-800 text-slate-200 border-slate-700 shadow-xl"
                                        style={{ maxWidth: "512px", width: "auto" }}
                                    >
                                        <h4 className="font-semibold text-white text-sm mb-1">{detail.label}</h4>
                                        <p className="text-xs leading-relaxed text-slate-300 whitespace-normal break-all">
                                            {detail.description}
                                        </p>
                                    </PopoverContent>
                                </Popover>
                            )}
                        </div>
                        <span className="text-white font-medium">{detail.value}</span>
                    </div>

                    {detail.image && (
                        <div className="mt-3">
                            <img src={detail.image} alt={detail.label} className="max-w-full rounded-lg border border-white/10" />
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}


