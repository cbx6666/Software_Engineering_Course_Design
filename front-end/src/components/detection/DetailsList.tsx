import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { HelpCircle } from "lucide-react";
import type { DetectionDetail } from "@/types/detection";

export function DetailsList({ details }: { details: DetectionDetail[] }) {
    const resultDetails = details.filter((d) => d != null);
    if (resultDetails.length === 0) return null;

    return (
        <div className="details-list">
            <h4>详细指标</h4>
            {resultDetails.map((detail, index) => (
                <div key={index} className="detail-row">
                    <div className="detail-label">
                            <span>{detail.label}</span>
                            {detail.description && (
                                <Popover>
                                    <PopoverTrigger asChild>
                                        <button className="transition-colors focus:outline-none">
                                            <HelpCircle className="w-4 h-4" />
                                        </button>
                                    </PopoverTrigger>
                                    <PopoverContent
                                        side="right"
                                        align="start"
                                        className="glass-popover"
                                        style={{ maxWidth: "512px", width: "auto" }}
                                    >
                                        <h4>{detail.label}</h4>
                                        <p>
                                            {detail.description}
                                        </p>
                                    </PopoverContent>
                                </Popover>
                            )}
                    </div>
                    <span className="detail-value">{detail.value}</span>

                    {detail.image && (
                        <div className="detail-image">
                            <img src={detail.image} alt={detail.label} />
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}


