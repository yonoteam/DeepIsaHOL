
signature SECTIONS =
sig
  type 'a T
  val theory_tag: string
  val thy_block_tag: string
  val proof_tag: string
  val other_tag: string
  val map: ('a -> 'b) -> 'a T -> 'b T
  val is_closed: 'a T -> bool
  val contains_tag: string -> 'a T -> bool
  val get_top_tag: 'a T -> string option
  val get_topmost: string -> 'a T -> 'a T list
  val get_first: 'a T -> 'a
  val get_last: 'a T -> 'a
  val get_latest_open: 'a T -> 'a T
  val next_state: ('a -> 'b -> 'b) -> 'a T -> 'b -> 'b
  val reverse: 'a T -> 'a T
  val recover_trace: 'a T -> 'a list
  val raw_segmentate: ('a -> string) -> (string * ('a -> bool) * ('a -> bool)) list 
    -> 'a list -> 'a T
end;


structure Sections: SECTIONS =
struct

(** Sections.T **)

datatype 'a T = 
    Empty
  | Single of 'a
  | Block of string * bool * ('a T list);

val theory_tag = "THEORY"
val thy_block_tag = "THY_BLOCK"
val proof_tag = "PROOF"
val other_tag = "OTHER"

(* map: ('a -> 'b) -> 'a Sections.T -> 'b Sections.T *)
fun map _ Empty = Empty
  | map f (Single a) = Single (f a)
  | map f (Block (tag, closed, sects)) = Block (tag, closed, List.map (map f) sects)

(* is_closed: 'a Sections.T -> bool *)
fun is_closed (Block (_, closed, sects)) = closed andalso forall is_closed sects
  | is_closed _ = true;

(* contains_tag: string -> Actions.section -> bool *)
fun contains_tag _ (Empty) = false
  | contains_tag _ (Single _) = false
  | contains_tag tag (Block (block_tag, _, sects)) = 
      if tag = block_tag then true
      else exists (contains_tag tag) sects;

(* get_top_tag: 'a Sections.T -> string option *)
fun get_top_tag Empty = NONE
  | get_top_tag (Single _) = NONE
  | get_top_tag (Block (tag, _, _)) = SOME tag

(* get_latest_open: 'a Sections.T -> 'a Sections.T *)
fun get_latest_open (Block (tag, closed, sects)) = 
      if null sects then raise Fail ("empty block at" ^ tag)
      else if forall is_closed sects then
        if not closed then Block (tag, closed, sects) else Empty
      else Ops.last (List.map get_latest_open sects)
  | get_latest_open _ = Empty;

(* get_topmost: string -> 'a Sections.T -> ('a Sections.T) list *)
fun get_topmost _ Empty = []
  | get_topmost _ (Single _) = []
  | get_topmost tag (block as Block (block_tag, _, sects)) =
      if null sects then raise Fail ("empty block " ^ tag)
      else if block_tag = tag then [block]
      else flat (fold (cons o get_topmost tag) (rev sects) [])

(* get_first: 'a Sections.T -> 'a *)
fun get_first Empty = raise Fail "get_first: Empty"
  | get_first (Single a) = a
  | get_first (Block (tag, _, sects)) = 
      if null sects then raise Fail ("empty block at" ^ tag)
      else get_first (hd sects)

(* get_last: 'a Sections.T -> 'a *)
fun get_last Empty = raise Fail "get_last: Empty"
  | get_last (Single a) = a
  | get_last (Block (tag, _, sects)) = 
      if null sects then raise Fail ("empty block at" ^ tag)
      else get_last (Ops.last sects)

(* next_state: ('a -> 'b -> 'b) -> 'a Sections.T -> 'b -> 'b *)
fun next_state _ Empty st = st
  | next_state next (Single a) st = next a st
  | next_state next (Block (_, _, sects)) st = fold (next_state next) sects st;

(* reverse: 'a Sections.T -> 'a Sections.T *)
fun reverse (Block (tag, closed, sects)) = Block (tag, closed, rev (List.map reverse sects))
  | reverse sect = sect;

(* recover_trace: 'a Sections.T -> 'a list *)
fun recover_trace Empty = []
  | recover_trace (Single a) = [a]
  | recover_trace (Block (_, _, sects)) = 
      fold (fn sect => fn sects => sects @ (recover_trace sect)) sects []

local

(* 'a Sections.T -> 'a Sections.T *)
fun close_latest (Block (tag, closed, sects)) = 
      if null sects then raise Fail ("empty block " ^ tag)
      else if not closed then
        if forall is_closed sects then Block (tag, true, sects)
        else Block (tag, closed, close_latest (hd sects) :: (tl sects))
      else raise Fail ("tried to close closed block" ^ tag)
  |  close_latest _ = raise Fail ("tried to close closed nonblock");

(* ('a -> string) -> 'a * string list * string list -> 'a Sections.T -> 'a Sections.T *)
fun add_new_section text_of (a, starts, ends) (Empty) = 
      if not (null starts) then Block (hd starts, false, [Single a])
      else if not (null ends) then raise Fail ("Unbalanced blocks on " ^ text_of a)
      else Block (other_tag, false, [Single a])
  | add_new_section text_of _ (Single a) = 
      raise Fail ("Wrong call add_new_section' on Single of" ^ text_of a)
  | add_new_section text_of (a, starts, ends) (Block (block_tag, false, sects)) =
      if null sects then 
        raise Fail ("Nil input in Block before " ^ text_of a)
      else if is_closed (hd sects) then
        if not (null starts) then Block (block_tag, false, (Block (hd starts, false, [Single a])) :: sects)
        else if not (null ends) then
          if member (op =) ends block_tag then Block (block_tag, true, Single a :: sects)
          else raise Fail ("Closing " ^ block_tag ^ " block with " ^ hd ends ^ " at " ^ text_of a)
        else Block (block_tag, false, Single a :: sects)
      else
        if not (null starts) andalso not (null ends) then
          if hd ends = the (get_top_tag (hd sects)) then 
            add_new_section text_of (a, starts, ends) (Block (block_tag, false, (close_latest (hd sects)) :: tl sects))
          else 
            Block (block_tag, false, (add_new_section text_of (a, starts, ends) (hd sects)) :: tl sects)
        else
          Block (block_tag, false, (add_new_section text_of (a, starts, ends) (hd sects)) :: tl sects)
  | add_new_section text_of (a, _, _) (Block (_, true, sects)) = 
      if null sects then raise Fail ("Null sections for input Block") 
      else raise Fail ("unbalanced blocks at " ^ text_of a);

(* 'a -> ('b * ('a -> bool) * ('a -> bool)) list -> 'a * 'b list * 'b list *)
fun make_tag_separators x [] = (x, [], [])
  | make_tag_separators x ((tag, starts, ends) :: tses) =
      if starts x andalso ends x then 
        (x, tag :: (#2 (make_tag_separators x tses)), tag :: (#3 (make_tag_separators x tses)))
      else if starts x then 
        (x, tag :: (#2 (make_tag_separators x tses)), #3 (make_tag_separators x tses))
      else if ends x then 
        (x, #2 (make_tag_separators x tses), tag :: (#3 (make_tag_separators x tses)))
      else make_tag_separators x tses;

in

(* ('a -> string) -> (string * ('a -> bool) * ('a -> bool)) list -> 'a list -> 'a Sections.T *)
fun raw_segmentate text_of separators xs =
  let
    fun iter x sect = add_new_section text_of (make_tag_separators x separators) sect;
    val rev_result = fold iter xs Empty;
  in reverse rev_result end;

end;


end;
