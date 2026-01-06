export function InstructionList({
  colorClassName,
  items,
}: {
  colorClassName: string;
  items: string[];
}) {
  return (
    <ul className="text-slate-300 space-y-2">
      {items.map((text, idx) => (
        <li key={idx} className="flex items-start gap-2">
          <span className={`${colorClassName} mt-1`}>•</span>
          <span>{text}</span>
        </li>
      ))}
    </ul>
  );
}


