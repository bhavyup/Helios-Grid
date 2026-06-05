import { createPortal } from "react-dom";

export function Portal({ children }: { children: React.ReactNode }) {
  // Check if we are in the browser before trying to access document
  if (typeof window === "undefined") return null;
  return createPortal(children, document.body);
}
