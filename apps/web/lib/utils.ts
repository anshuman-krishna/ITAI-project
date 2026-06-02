import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

// merge conditional class names and resolve tailwind conflicts.
// this is the helper shadcn/ui components expect.
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
