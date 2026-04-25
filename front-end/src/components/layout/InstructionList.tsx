export function InstructionList({
  colorClassName,
  items,
}: {
  colorClassName: string;
  items: string[];
}) {
  return (
    <ul className="instruction-list">
      {items.map((text, idx) => (
        <li key={idx}>
          <span className={`instruction-list__mark ${colorClassName}`}>{idx + 1}</span>
          <span>{text}</span>
        </li>
      ))}
    </ul>
  );
}


